# Time Series Momentum

---

## Data

All data stored in 'data' folder. Only 'Return.xlsx' is used.

Check Return.xlsx, it contains returns of totally 55 assets.

## Model

There are two parts of model: deep learning & other methods (including AutoEncoder, PCA, PLS, OLS, Lasso regression, Ridge regression, Elastic Net ,Boosted regression Tree, and Random Forrest). These two parts are in different  folder, yet in the same structure.

*Here I will take './deep_learning/' folder as an example*

### Main folder [5 files totally]

​	"../deep_learning/code" folder is the template code that can fit in different type of data. If any change needed to be made, it can be made in the "code" folder, and code for all different data would be change.

* **main.py** - is the main python file, running result for one test year.

- **tuning.py** - tuning parameter of NN [batch size,...]

**Function py files**

 *	*	**single_run.py** - generate all results
      	   	*	**dl_function_ts.py** - store all NN function

   	

## How to run

1. Run "./create_folder_and_submit.py"

   This will generate a folder for each asset based on the template **"code"** folder and two files: "submit_jobGen.sh"  and "submit.sh" , and also under folder for  each asset, "asset1" for example, Generate python file for each test year  as **script{test_year}.py** based on main.py and shell file for summit based on **deeplearning_stock.sh**

   ***NOTE!!! change the path of old folder in line 17 with your path of "code" folder***

3. Now upload the whole 'deep_learning' folder to Mercury.

4. After all files uploaded. "sh submit.sh" in './deep_learning' and all job summit.

Refer to this folder in Mercury "/project/polson/GavinFeng/gaomin/TSM/0808/"



