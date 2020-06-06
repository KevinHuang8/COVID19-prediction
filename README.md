# COVID-19 Prediction Models

`models` contains the code for all of our models, and the functions for loading/processing the data

- `models.curvefit` is the curve fitting model that used for the final submission
- `models.gaussianprocess` is the "Unincorporated model" Gaussian process model described in the report
- `models.LSTM`  is the LSTM model used
- `models.utils.dataloader` contains all of the functions used for loading various types of data

`notebooks` contains various jupyter notebooks used for prototyping and testing models and ideas, though it is undocumented

`predictions` contains all model predictions

Example usage can be found in `Global_Minimum_Gang_16_code_demo.ipynb`

Model details can be found in `report.pdf`