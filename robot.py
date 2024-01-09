import re
from langchain.prompts import PromptTemplate
from utils.llm import llm

ten_by_ten_grid = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
]

def check_cell(x, y):
    return ten_by_ten_grid[y][x]

prompt_template = PromptTemplate.from_template("""
You are a robot navigating a {size}x{size} grid of cells. 
### Constraints ###
1. You can only move to adjacent cells.
2. You can check a cell by providing its coordinates. Eg [1,4].
3. Don't explain your moves.
4. Coordinates must stay within the bounds of the {size}x{size} grid. They start at 0.

### Instruction ###
find a cell with symbol {target}. 
Follow this patter:
Observation: the result of the previous move.
Thought: you should think about what to do next.
Move: the next cell to move to, for example [0,1]                                                              

{history}
""")

def extract_numbers(string):
    x = string.split("Move:")
    numbers = re.findall(r'\d+', x[1])
    return [int(number) for number in numbers]

target = 17
start = [5, 5]

history = f"""
Observation: I don't have any observations about cells yet.
Thought: My target is {target}, I don't have observations about other cells yet, so I'll move to [{start[0]}, {start[1]}].
"""

while True:
    prompt = prompt_template.format(size=10, target=target, history=history)
    output = llm.invoke(prompt, stop=["Observation:"])
    history += f"\n{output.strip()}"
    coordinates = extract_numbers(output)
    symbol = check_cell(coordinates[0], coordinates[1])
    history += f"\nObservation: I am at [{coordinates[0]}, {coordinates[1]}], it has symbol {symbol}."
    if symbol == target:
        print("You found the target!")
        print(history)
        break