import sys
import PyPDF2
import requests
import openai
import os
import time
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, logging
from tqdm import tqdm
import warnings

# Suppress the transformers library warnings
logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning, module='PyPDF2')

# Load .env file
load_dotenv()

# Access your API key
API_KEY = os.getenv('API_KEY')

# Define the cost per token
TOKEN_RATE = 0.0200 / 1000  # 0.02 dollars per 1K tokens, adjust as needed


openai.api_key = API_KEY

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def chat_with_gpt(prompt, max_completion_tokens=1000):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_completion_tokens,
            n=1,
            temperature=1.0,
            stop=None,
        )

        if response.choices:
            return response.choices[0].text
        else:
            print("Error: No response generated.")
            return None
    except openai.error.RateLimitError:
        print("Error: Rate limit exceeded. Please try again later.")
        return None


def read_pdf_from_url(pdf_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "application/pdf",
        "If-Range": "{497B36C3-2D45-42EA-B30A-8A191BA6F53C},1",
        "Range": "bytes=0-",
        "Upgrade-Insecure-Requests": "1"
    }
    response = requests.get(pdf_url, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    if total_size == 0:
        print("Downloading PDF of unknown size, please wait...", end="", flush=True)
        with open('pdf_file.pdf', 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        print(" Done")
    else:
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open('pdf_file.pdf', 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

    print("\nAnalyzing document, please wait...", end="", flush=True)
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)

    with open('pdf_file.pdf', "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        num_pages = reader.getNumPages()
        text = ""
        for i in range(num_pages):
            page = reader.getPage(i)
            page_text = page.extractText()
            if page_text:
                text += page_text

        if not text:
            print("\nError: The PDF contains no text or only scanned images.")
            sys.exit(1)

        return text

def count_tokens(text):
    return len(tokenizer.encode(text))


def split_into_chunks(text, max_tokens=4096):
    words = text.split()
    chunks = []

    current_chunk = []
    current_chunk_tokens = 0

    for word in words:
        token_length = count_tokens(word)

        if current_chunk_tokens + token_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0

        current_chunk.append(word)
        current_chunk_tokens += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_pdf_chunks(chunks, question, user_feedback=None):
    total_tokens = 0

    context = build_context_within_limit(chunks, question, user_feedback)

    system_message = f"System: You are an AI language model, and your task is to answer questions about the following document. The document contains {len(chunks)} chunks of text."
    context.insert(0, system_message)

    max_context_length = 4096
    available_tokens = max_context_length - count_tokens("\n".join(context) + f"\n{question}")

    print("Context tokens:", count_tokens("\n".join(context)))
    print("Question tokens:", count_tokens(question))
    print("Available tokens:", available_tokens)

    if available_tokens > 50:
        max_tokens = min(available_tokens, 1024)

        print("Max tokens:", max_tokens)

        response = chat_with_gpt("\n".join(context) + f"\n{question}", max_completion_tokens=max_tokens)

        if response:
            total_tokens += count_tokens(response)
            return response.strip(), total_tokens
    else:
        return "I'm sorry, but there is too much context for me to generate a meaningful answer. Please try asking a more specific question or reduce the amount of context.", total_tokens


def build_context_within_limit(chunks, question, user_feedback=None, max_tokens=4096):
    context = []
    question_tokens = count_tokens(question)
    completion_tokens = 1024
    buffer_tokens = 20
    available_tokens = max_tokens - (question_tokens + completion_tokens + buffer_tokens)

    if user_feedback:
        available_tokens -= count_tokens(user_feedback)
        context.append(user_feedback)

    for chunk in reversed(chunks):
        chunk_tokens = count_tokens(chunk)
        if available_tokens - chunk_tokens > 0:
            context.insert(0, chunk)
            available_tokens -= chunk_tokens
        else:
            break

    return context


def calculate_cost(tokens_used, rate_per_token):
    return tokens_used * rate_per_token


def get_user_feedback():
    while True:
        feedback = input("Rate the answer (^/v/Spacebar): ").strip().lower()
        if feedback in ["^", "v", ""]:
            return feedback
        else:
            print("Invalid input. Please enter '^', 'v', or press the Spacebar for bypass.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 chat.py <URL>")
        sys.exit(1)

    pdf_url = sys.argv[1]

    try:
        text = read_pdf_from_url(pdf_url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the PDF: {e}")
        sys.exit(1)

    chunks = split_into_chunks(text)

    try:
        num_chunks = len(chunks)
        avg_chunk_length = len(text) / num_chunks
        avg_tokens_per_chunk = count_tokens(text) / num_chunks
    except ZeroDivisionError:
        print("Error: The PDF contains no text or only scanned images.")
        sys.exit(1)

    print("\nInteractive chat with the PDF has started. Type 'quit' to end the chat.\n")
    total_tokens = 0
    user_feedback = None

    try:
        while True:
            question = input("Ask a question: ")

            if question.lower() == 'quit':
                print("Ending chat.")
                break

            response, tokens_used = process_pdf_chunks(chunks, question, user_feedback)
            total_tokens += tokens_used
            print(response)
            print()

            feedback = get_user_feedback()

            if feedback == "v":
                user_feedback = input("Please provide the correct information: ")
            elif feedback == "^":
                user_feedback = None
            # No need for an action for the 'bypass' option (Spacebar)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Ending chat.")

    total_cost = calculate_cost(total_tokens, TOKEN_RATE)

    print(f"Total tokens used during the session: {total_tokens}")
    print(f"Total cost of API calls during the session: {total_cost}")


if __name__ == "__main__":
    main()
