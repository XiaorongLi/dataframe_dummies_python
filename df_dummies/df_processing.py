import numpy as np
import pandas as pd


class dfCategorical:
    def __init__(self, df, response):
        """ dfCategorical class for converting data with categorical variables
        into matrix with properly encoded dummy variables. If requested, missing
        values in numerical columns can be imputed.
        
        Attributes:
            df (pandas dataframe), needs to be processed for succeeding modelling
            respongse (string), variable name of the response in the model
            num_vars (list), names of numerical variables
            cat_vars (list), names of categorical variables
            no_res_df (dataframe), dataframe without the response column
            dummy_df (pandas dataframe), result dataframe after processing
        """
        
        self.df = df
        self.response = response
        self.num_vars = self.datatype_identify()[0]
        self.cat_vars = self.datatype_identify()[1]
        self.no_res_df = self.datatype_identify()[2]
        self.imputed_df = self.imputation()
        
        
        
    def datatype_identify(self):
        """Function to remove the response variable and then identify the data type
        for the rest variables.
        
        Args: none
        Return: a list containing two lists and a dataframe: 
        1)categorical variable names
        2)numerical variable names
        3)dataframe with response column removed
        """
        
        #drop rows with missing respons
        df = self.df
        no_res_df = df.dropna(subset=[self.response], axis=0)
        
        num_vars = list(no_res_df.select_dtypes(include=['float', 'int']).columns)
        cat_vars = list(no_res_df.select_dtypes(include=['object']).columns)
        
        return [num_vars, cat_vars, no_res_df]
    
    def imputation(self, method = "mean"):
        """Function to impust the missing values of numerical variables. Currently
        column mean and column median are supported.
        
        Arg: method, indicating imputed values, mean or median
        return: imputed dataframe (no response) variable if there are missing values; 
        otherwise return the original dataframe (no response)
        """
        
        no_res = self.datatype_identify()[2]
        #no_res = self.no_res_df #THIS IS PROBLEMATIC!!! Seems self.no_res_df gets changed
                                 #by the imputation method!!! Need to figure out why.
        if no_res.isnull().sum().sum() == 0:
            return no_res
        else:
            if method == "mean":                
                for col in self.num_vars:
                    no_res[col].fillna(no_res[col].mean(), inplace=True)
            if method == "median":
                for col in self.num_vars:
                    no_res[col].fillna(no_res[col].median(), inplace=True)
            imputed_df = no_res
            return imputed_df
        
    def encode_dummies(self, dummy_na = True, impute=True):
        """Function that encodes all the categorical variables into dummy variables,
        using new variable names with underscore '_'. The first level of the
        categorical value is by default dropped. If requested, the missing values 
        in the numerical variables are imputed.
        
        Arg: 
        dummy_na, if True, encode an additional dummy variable for NAs;
        otherwise encode NAs as 0's.
        impute, if True, missing numerical values are imputed.
        
        """
        if impute:
            mydf = self.imputation()
        else:
            mydf = self.datatype_identify()[2]
        
        for col in self.cat_vars:
            mydf = pd.concat([mydf.drop(col, axis=1),\
                                  pd.get_dummies(mydf[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis = 1)
        dummy_df = mydf
        return dummy_df
        