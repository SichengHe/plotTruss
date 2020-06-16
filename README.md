# Purpose

Visualize a truss structure with its buckling and stress constraints.
This repo was created for the paper [A novel approach to discrete truss design problems using mixed integer neighborhood search](https://link.springer.com/article/10.1007/s00158-018-2099-8).

# How to use

## Tutorial
For a tutorial run, simply type.

```
python plotTruss.py
```

Six figures of stress and buckling constraints will be generated.

## More advanced
If you need to change the data, then follow the following instruction.
Put the element, nodes and solution files under `INPUT` directory.
Modify the lines in `plotTruss.py` for `filename_node, filename_elem, filename_con, filename_sol`. 
Usually in the solution files, the element cross section area and nodal displacement are placed together. 
It requires the user to input the corresponding lines for these pieces of information by changing `ind_beg_elem, ind_end_elem` which corresponds with element information and `ind_beg_node, ind_end_node` which correspond with the nodal information.

Also, notice that the user shall include the fixed nodes in the displacement data.

Further, if colorbars are needed, please refer to the `/other`.

To truncated additional white space in `.pdf`, use `pdfcrop` command.
