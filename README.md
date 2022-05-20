I wanted to explore making PyTorch models from scratch. 

Many tutorials ignore tabular data or if tabular data is used there are layers of abstraction which I do not find useful as someone wanting to research various NN architectures as a hobby.

This is a reference project that contains bite sized sample scripts.

This most interesting scripts are those which predict y using a polynomial x as an input.


$$ y = x + x^2 + x^3 + e $$


Here is what the raw data looks like

![data](data_gen_and_plot/plots/data.jpg)

Here is a regression line fit through the data

![data](data_gen_and_plot/plots/regression_fit.jpg)

Here is a regression line fit with a PyTorch line added (dotted red)

![data](data_gen_and_plot/plots/py_torch_regression_fit.jpg)

Finally, I like to make sure I can get back to some statistics. I add confidence intervales here. I did this simply by

$$ \hat{y} \pm [1.96, 1.00]$$

![data](data_gen_and_plot/plots/py_torch_regression_fit_ci.jpg)

