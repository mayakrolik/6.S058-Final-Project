from openai import OpenAI
from KEYS import OPENAI_KEY

client = OpenAI(
  api_key=OPENAI_KEY
)


def call_chat(prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content


def iterative_prompt(purpose, missing_cat):
    """
    create a nice prompt
    """
    return f"""You are a data scientist trying to create a ballanced dataset for {purpose}. You notice that you are missing data points of {missing_cat} related to the topic.
    
    What search queary would you reccomend inputting into a source like YouTube to find data related to {purpose} and involving {missing_cat}? 
    
    Feel free to be creative and name specific people or groups in your reccomendation as that will help make the youtube search more accurate. Please also ensure that your reccomendation encourages the search of real people and not animation or voice over content. Note that explicitly stating the desired data science goal might not be a good representation of how to find the desired data online as people do not explicitly label their posts pragmatically.
    
    Please ONLY output the ONE reccomended search terms with NO justification of why you selected it. Thanks!"""

def initial_promt(purpose):
    """
    create a nice initial prompt given a dataset purpose
    """

    return f"""You are a data scientist trying to create a ballanced dataset for {purpose}.
    
    What search queary would you reccomend inputting into a source like YouTube to find data related to {purpose} to give a good demographic spread of results? 
    
    Feel free to be creative and name specific people, groups, and/or channels in your reccomendation as that will help make the youtube search more accurate. Please also ensure that your reccomendation encourages the search of real people and not animation or voice over content. Note that explicitly stating the desired data science goal might not be a good representation of how to find the desired data online as people do not explicitly label their posts pragmatically.
    
    Please ONLY output the ONE reccomended search terms with NO justification of why you selected it. Thanks!"""

if __name__ == "__main__":
    purp = "recognizing lip motion in English"
    cat_missing = "asian females old"
    
    prompt = iterative_prompt(purp, cat_missing)
    res = call_chat(prompt)
    print(res)