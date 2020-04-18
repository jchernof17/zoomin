# Our notes

## `generator.py`
In `generator.py` I put some functions that can generate and save random graphs. This file can be run normally, or you can copy/paste the code and put it into a `.ipynb` and use the matplotlib package to actually visualize the graphs. An example of that is in the `generate_tree` function.

This file was also used to generate the `.in` files. They're in the repo right now, but keep in mind if you run `generator.py` and then commit changes elsewhere, you're going to overwrite the `.in` files. It'sfine but just kinda annoying. Maybe don't stage those files for commit.


# Official Spec

Take a look at the project spec before you get started!

Requirements:

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions
