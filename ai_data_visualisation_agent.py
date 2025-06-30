import os
import sys
import re
import io
import json
import base64
import contextlib
import warnings
from typing import Optional, List, Any, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox

# Suppress unnecessary pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Regex pattern to extract code blocks from LLM responses
pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    """Run Python code in a secure E2B sandbox environment and return results."""
    with st.spinner("Executing code in E2B sandbox..."):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None

        return exec.results


def match_code_blocks(llm_response: str) -> str:
    """Extract Python code block from LLM response."""
    match = pattern.search(llm_response)
    return match.group(1) if match else ""


def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    """Send user query and dataset context to the LLM and return code results + raw response."""
    system_prompt = f"""You're a Python data scientist and data visualization expert.
You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner("Getting response from Together AI LLM model..."):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)

        if python_code:
            code_results = code_interpret(e2b_code_interpreter, python_code)
            return code_results, response_message.content
        else:
            st.warning("No Python code block detected in LLM response.")
            return None, response_message.content


def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    """Upload file to sandbox and return path."""
    dataset_path = f"./{uploaded_file.name}"
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise


def main():
    """Main Streamlit app entry point."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Session state initialization
    st.session_state.setdefault("together_api_key", "")
    st.session_state.setdefault("e2b_api_key", "")
    st.session_state.setdefault("model_name", "")

    with st.sidebar:
        st.header("API Keys and Model Configuration")

        st.session_state.together_api_key = st.text_input("Together AI API Key", type="password")
        st.sidebar.info("💡 Everyone gets a free $1 credit by Together AI")
        st.markdown("[Get Together AI API Key](https://api.together.ai/signin)")

        st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password")
        st.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")

        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }

        selected_model = st.selectbox("Select Model", options=list(model_options.keys()), index=0)
        st.session_state.model_name = model_options[selected_model]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")

        if st.checkbox("Show full dataset"):
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())

        query = st.text_area("What would you like to know about your data?",
                             "Can you compare the average cost for two people between different categories?")

        if st.button("Analyze"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)

                    st.subheader("AI Response:")
                    st.write(llm_response)

                    if code_results:
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:
                                png_data = base64.b64decode(result.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=False)
                            elif hasattr(result, 'figure'):
                                st.pyplot(result.figure)
                            elif hasattr(result, 'show'):
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)


if __name__ == "__main__":
    main()
