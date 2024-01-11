from langchain.prompts import PromptTemplate
from infra.llm import llm


# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# key_points_prompt = PromptTemplate.from_template("""
# You are a movie script writer. Write key plot points and themes that sequel of the movie {movie} must have. Don't write the actual script.
# Plot points and themes:
# """)

# script_prompt = PromptTemplate.from_template("""
# You are a movie script writer. Write a script for the sequel of the movie {movie}. The script must include the following plot points.
# Plot points: {plot_points}

# Script:
# """)

# sequel_chain = {
#     "movie": RunnablePassthrough(),
#     "plot_points": key_points_prompt | llm,
# } | script_prompt | llm

# # sequel_chain.invoke({
# #     "movie": "inception"
# # })

prompt = PromptTemplate.from_template("""
You are a high school history teacher grading homework assignments.
Based on the question and the correct answer, your task is to determine whether the student's answer is
correct.
Grading is binary; therefore, student answers can be correct or wrong.
Simple misspellings are okay. No explanation needed.
                                      
### examples ###
Question: what is 2+2?
Correct Answer: 4
Student Answer: 5
Verdict: Wrong

Question: {question}
Correct Answer: {correct_answer}
Student Answer: {student_answer}
Verdict:
""")


student_answer_list = ["John,F. Kennedy", "JFK", "FDR", "John F. Kenedy",
"John Kennedy", "Jack Kennedy", "Jacquelin Kennedy", "Robert F. Kenedy"]

questions = [
    {
        "question": "What is the name of the 35th president of the United States?",
        "correct_answer": "John F. Kenedy",
    },
]

teacher_chain = prompt | llm.bind(stop=["Question:"])


for answer in student_answer_list:
    teacher_chain.invoke({  
        'question':questions[0]['question'], 
        'correct_answer':questions[0]['correct_answer'],
        'student_answer': answer,
    })


