import argparse
import re

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import javalang

from prompts import system_message, system_message2

# =====================================================
# 1. Hyperparams & Device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN_CODE = 64
MAX_LEN_COMMENT = 256
MAX_LEN_TOTAL = MAX_LEN_CODE + MAX_LEN_COMMENT


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with optional LLM calls.")
    parser.add_argument("--api_key", type=str, help="API key for OpenAI or other LLMs")
    return parser.parse_args()


# -----------------------------------------------------
# 2. Subtokenization Helpers
# -----------------------------------------------------
def subtokenize_identifier(identifier):
    temp = identifier.replace("_", " ")
    temp = re.sub(r'([a-z])([A-Z])', r'\1 \2', temp)
    tokens = re.split(r'([^a-zA-Z0-9]+)', temp)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


def subtokenize_sequence(token_list):
    all_subtokens = []
    for token in token_list:
        subtoks = subtokenize_identifier(token)
        all_subtokens.extend(subtoks)
    return all_subtokens


# -----------------------------------------------------
# 3. Separate Javadoc and Code
# -----------------------------------------------------
def separate_javadoc_and_code(input_text):
    pattern = re.compile(r'/\*\*(.*?)\*/', re.DOTALL)
    match = re.search(pattern, input_text)
    if match:
        javadoc_text = match.group(1).strip()
        code_text = input_text.replace(match.group(0), '')
    else:
        javadoc_text = ""
        code_text = input_text
    return javadoc_text, code_text


# -----------------------------------------------------
# 4. Extract Javadoc Sections (ignore @throws)
# -----------------------------------------------------
def extract_javadoc_sections(javadoc_text):
    lines = javadoc_text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.lstrip(" *\t/")  # remove leading * or spaces
        cleaned_lines.append(line)

    summary_lines = []
    param_blocks = []
    return_blocks = []

    current_section = "summary"
    current_param = []
    current_return = []
    ignoring_throws = False

    for line in cleaned_lines:
        if line.startswith("@param"):
            ignoring_throws = False

            if current_section == "param" and current_param:
                param_blocks.append(current_param)
            if current_section == "return" and current_return:
                return_blocks.append(current_return)
            current_section = "param"
            current_param = []
            param_line = line.replace("@param", "").strip()
            current_param.append(param_line)

        elif line.startswith("@return"):
            ignoring_throws = False
            # finalize old block
            if current_section == "param" and current_param:
                param_blocks.append(current_param)
            if current_section == "return" and current_return:
                return_blocks.append(current_return)
            current_section = "return"
            current_return = []
            return_line = line.replace("@return", "").strip()
            current_return.append(return_line)

        elif line.startswith("@throws"):
            ignoring_throws = True

            if current_section == "param" and current_param:
                param_blocks.append(current_param)
            if current_section == "return" and current_return:
                return_blocks.append(current_return)
            current_section = "throws"

        else:
            if ignoring_throws:
                continue
            if current_section == "summary":
                summary_lines.append(line)
            elif current_section == "param":
                current_param.append(line)
            elif current_section == "return":
                current_return.append(line)
            else:
                # throws section => ignore
                pass

    if current_section == "param" and current_param:
        param_blocks.append(current_param)
    if current_section == "return" and current_return:
        return_blocks.append(current_return)

    summary_text = " ".join(summary_lines).strip()
    param_texts = [" ".join(block).strip() for block in param_blocks]
    return_texts = [" ".join(block).strip() for block in return_blocks]

    return summary_text, param_texts, return_texts


# -----------------------------------------------------
# 5. Code Parsing with javalang
# -----------------------------------------------------
def parse_code_with_javalang(code_text):
    try:
        tokens = list(javalang.tokenizer.tokenize(code_text))
    except:
        naive_code_tokens = re.findall(r'[A-Za-z0-9_]+|\S', code_text)
        return subtokenize_sequence(naive_code_tokens)

    token_strings = [t.value for t in tokens]
    code_subtokens = subtokenize_sequence(token_strings)
    return code_subtokens


# -----------------------------------------------------
# 6. Combine comment + code
# -----------------------------------------------------
def combine_for_inference(comment_tokens, code_tokens, tokenizer):
    comment_str = " ".join(comment_tokens)
    code_str = " ".join(code_tokens)
    combined_input = f"{comment_str} {tokenizer.sep_token} {code_str}"

    encoded = tokenizer(
        combined_input,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_TOTAL,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


# -----------------------------------------------------
# 7. Single-section Inference
# -----------------------------------------------------
def run_inference_for_section(model, tokenizer, comment_tokens, code_tokens):
    input_ids, attention_mask = combine_for_inference(comment_tokens, code_tokens, tokenizer)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        prob = torch.sigmoid(logits).item()
    return prob


# -----------------------------------------------------
# 8. Main Inference
# -----------------------------------------------------
def main_inference(input_text):
    javadoc_text, code_text = separate_javadoc_and_code(input_text)
    summary_text, param_texts, return_texts = extract_javadoc_sections(javadoc_text)

    summary_tokens = subtokenize_sequence(summary_text.split())
    param_tokens_list = [subtokenize_sequence(p.split()) for p in param_texts]
    return_tokens_list = [subtokenize_sequence(r.split()) for r in return_texts]

    code_tokens = parse_code_with_javalang(code_text)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    model_summary = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1
    )
    model_summary.load_state_dict(torch.load("default_head_models/Summary_best_model.pt", map_location=DEVICE))
    model_summary.to(DEVICE)
    model_summary.eval()

    model_param = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1
    )
    model_param.load_state_dict(torch.load("default_head_models/Param_best_model.pt", map_location=DEVICE))
    model_param.to(DEVICE)
    model_param.eval()

    model_return = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1
    )
    model_return.load_state_dict(torch.load("default_head_models/Return_best_model.pt", map_location=DEVICE))
    model_return.to(DEVICE)
    model_return.eval()

    # Summary inference
    summary_prob = 0.0
    summary_label = None
    if summary_tokens:
        summary_prob = run_inference_for_section(model_summary, tokenizer, summary_tokens, code_tokens)
        summary_label = 1 if summary_prob > 0.5 else 0

    # Param inference
    param_probs = []
    param_labels = []
    for param_tokens in param_tokens_list:
        param_prob = run_inference_for_section(model_param, tokenizer, param_tokens, code_tokens)
        param_label = 1 if param_prob > 0.5 else 0
        param_probs.append(param_prob)
        param_labels.append(param_label)

    # Return inference
    return_probs = []
    return_labels = []
    for ret_tokens in return_tokens_list:
        ret_prob = run_inference_for_section(model_return, tokenizer, ret_tokens, code_tokens)
        ret_label = 1 if ret_prob > 0.5 else 0
        return_probs.append(ret_prob)
        return_labels.append(ret_label)

    results = {
        "summary": {
            "text": summary_text,
            "prob": summary_prob,
            "label": summary_label
        },
        "params": [
            {"text": ptxt, "prob": pprob, "label": plabel}
            for (ptxt, pprob, plabel) in zip(param_texts, param_probs, param_labels)
        ],
        "returns": [
            {"text": rtxt, "prob": rprob, "label": rlabel}
            for (rtxt, rprob, rlabel) in zip(return_texts, return_probs, return_labels)
        ],
    }
    return results


# -----------------------------------------------------
# 9. ChatGPT calls: openai_call
# -----------------------------------------------------
def openai_call(prompt, prompt2, api_key):
    if not api_key:
        st.write("LLM responses cannot be taken since the API key is not provided.")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # With model probabilities in the prompt
        completion_with_probs = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message2},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.0,
        )
        st.write("==== LLM Response (with model probabilities) ====")
        st.write(completion_with_probs.choices[0].message.content)

        # Without model probabilities in the prompt
        completion_no_probs = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        st.write("==== LLM Response (without model probabilities) ====")
        st.write(completion_no_probs.choices[0].message.content)

    except ImportError:
        st.write("OpenAI library is not installed. Please install it to use LLM calls.")
    except Exception as e:
        st.write(f"An error occurred while calling the LLM: {str(e)}")


# -----------------------------------------------------
# 10. Streamlit UI
# -----------------------------------------------------
def highlight_block(text_block, label):
    if label == 1:
        # Inconsistent
        background_color = "#e74c3c"
        text_color = "#ffffff"
    else:
        # Consistent
        background_color = "#2ecc71"
        text_color = "#000000"

    highlighted = (
        f"<div style='background-color: {background_color}; "
        f"color: {text_color}; padding: 0.5em; border-radius: 5px;'>"
        f"{text_block}</div>"
    )
    return highlighted


def main():
    args = get_args()
    api_key = args.api_key

    st.title("JINSIGHT - Javadoc Consistency Checker")
    default_javadoc_code = """
        /**
         * Calculates the average of numbers in a list.
         *
         * @param numberList a list of numbers
         * @return the average of the integers in the list
         */
        public static double calculateAverage(List<Integer> numbers) {
            return numbers.stream()
                    .mapToInt(Integer::intValue)
                    .average()
                    .orElse(0.0);
        }
        """

    user_input = st.text_area("Javadoc + Code Input:", value=default_javadoc_code, height=300)

    if st.button("Run Inference + Call LLM"):
        with st.spinner("Running inference..."):
            results = main_inference(user_input)

            summary_text = results["summary"]["text"]
            summary_prob = results["summary"]["prob"]
            summary_label = results["summary"]["label"]

            st.markdown("### Summary Section")
            st.write(f"Probability: {summary_prob:.4f} → Label: {summary_label}")
            st.markdown(highlight_block(summary_text, summary_label), unsafe_allow_html=True)

            if results["params"]:
                st.markdown("### Param Sections")
                for i, p in enumerate(results["params"], start=1):
                    st.write(f"**Param {i}** Probability: {p['prob']:.4f} → Label: {p['label']}")
                    st.markdown(highlight_block(p['text'], p['label']), unsafe_allow_html=True)

            if results["returns"]:
                st.markdown("### Return Sections")
                for i, r in enumerate(results["returns"], start=1):
                    st.write(f"**Return {i}** Probability: {r['prob']:.4f} → Label: {r['label']}")
                    st.markdown(highlight_block(r['text'], r['label']), unsafe_allow_html=True)

            prompt_without_probs = (
                "Below is the Javadoc + Code. "
                f"{user_input}\n\n"
                "Output:"
                "\nEND"
            )

            summary_msg = f"Summary label={summary_label} prob={summary_prob:.4f}"
            param_msg = "\n".join([
                f"Param {i + 1} label={p['label']} prob={p['prob']:.4f}"
                for i, p in enumerate(results["params"])
            ])
            return_msg = "\n".join([
                f"Return {i + 1} label={r['label']} prob={r['prob']:.4f}"
                for i, r in enumerate(results["returns"])
            ])
            prompt_with_probs = (
                "Below is the Javadoc + Code AND model predictions: "
                f"{user_input}\n\n"
                "Model Predictions:\n"
                f"{summary_msg}\n{param_msg}\n{return_msg}\n\n"
                "Output:"
                "\nEND"
            )

        with st.spinner("Getting LLM Responses..."):
            st.markdown("### Model Responses")
            openai_call(prompt_without_probs, prompt_with_probs, api_key)


if __name__ == "__main__":
    main()
