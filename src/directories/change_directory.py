#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path


# In[2]:


class ChangeDirectory:
    def __init__(self):
        self.MAIN_DIR = os.getcwd()
        
    def change_to_src_models_dir(self):
        parent_path = Path(self.MAIN_DIR).parent
        data_path = str(parent_path) + "/src/models"
        os.chdir(data_path)
        
    def change_to_data_dir(self):
        parent_path = Path(self.MAIN_DIR).parent
        data_path = str(parent_path) + "/data"
        os.chdir(data_path)
        
    def change_to_model_dir(self):
        parent_path = Path(self.MAIN_DIR).parent
        data_path = str(parent_path) + "/models"
        os.chdir(data_path)
        
    def change_to_notebook_dir(self):
        os.chdir(self.MAIN_DIR)

