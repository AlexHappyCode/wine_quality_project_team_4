import sys
import nbformat

first_notebook = sys.argv[1]
second_notebook = sys.argv[2]

# Reading the notebooks
first_notebook = nbformat.read(first_notebook, 4)
second_notebook = nbformat.read(second_notebook, 4)

# Creating a new notebook
final_notebook = nbformat.v4.new_notebook(metadata=first_notebook.metadata)

# Concatenating the notebooks
final_notebook.cells = first_notebook.cells + second_notebook.cells

# Saving the new notebook 
nbformat.write(final_notebook, 'final_notebook.ipynb')
