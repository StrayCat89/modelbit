import modelbit, sys, os

# Secrets automatically included by Modelbit
os.environ["OPENAI_API_KEY"] = modelbit.get_secret("OPENAI_API_KEY", ignore_missing=True)

from openai import OpenAI

prompt_header = modelbit.load_value("data/prompt_header.pkl") #  Summarize the following product reviews. The reviews are all about the same product. Be as concise as possible. Your summarized review should include the most important details about the product's qu...

# main function
def summarize_product_reviews(reviews: str):
    prompt = prompt_header + "\n\n" + reviews

    completion = OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])

    summary = completion.choices[0].message.content
    return { "summary": summary }

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = summarize_product_reviews(...)
#   print(result)