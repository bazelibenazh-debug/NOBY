 notebook description
This notebook, carbon_bombs_analysis.ipynb, looks at large fossil fuel extraction projects known as carbon bombs. These are oil, gas, or coal projects that could each release more than one gigaton of CO₂ if fully developed. Because of their size, these projects are especially important when discussing climate change and the goals of the Paris Agreement.
The notebook starts by loading the Carbon Bombs dataset from Kühne et al. (2022). The data are cleaned and simplified so that only the most useful information is kept. This includes the project name, country, fossil fuel type, whether the project is new, and its estimated CO₂ emissions. The cleaned data are stored in a dataset called ds_carbon_bombs, with basic metadata added to explain what each variable represents and which units are used.
Next, the notebook explores the data using plots and summaries. It focuses on the 20 largest carbon bomb projects based on their potential emissions. Bar plots are used to show these projects and to compare them by country and fuel type. Other figures help show how emissions are spread across different countries and fossil fuels, and how much these projects could contribute to future emissions overall.
The analysis is then extended using two additional datasets related to the Permian Delaware Tight carbon bomb. These datasets estimate how many children born in 2020 are expected to face more climate extremes because of emissions from this project. The notebook uses maps to show these impacts at regional and country scales, making it easier to see where the risks are highest.
Because these estimates are uncertain, the notebook also includes 90% confidence intervals. The plots show best estimates together with their uncertainty ranges for each region and for the most affected countries. Belgium is included as a comparison. These results show both the size of the expected impacts and how uncertain the projections are.
Finally, the notebook is organized so that each step of the analysis is easy to follow. The main data processing and plotting steps are later turned into reusable functions and moved into a separate Python script, carbon_bombs_analysis.py, which runs the full analysis in a clean and structured way.

References:

Kühne, K., et al. (2022). Gas, oil, and coal projects consistent with limiting global warming to 1.5°C. Energy Policy, 173, 113279. https://doi.org/10.1016/j.enpol.2022.113279
Additional datasets used in this analysis, including climate impact projections and geospatial boundary files, are included in the project directory.

