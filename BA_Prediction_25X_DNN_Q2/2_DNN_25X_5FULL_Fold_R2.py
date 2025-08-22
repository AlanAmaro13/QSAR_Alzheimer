#!/usr/bin/env python
# coding: utf-8

# # Dense Neural Networks

# * This notebook MUST BE restart for each fold for a clean start in Keras weights.

# ## Mutual Information

# ### **Mutual Information (MI) for Feature Selection**  
# **Mutual Information** measures the **statistical dependence** between two variables, capturing **both linear and nonlinear relationships**. Unlike Pearson correlation (which only detects linear trends), MI quantifies how much knowing one variable reduces uncertainty about the other.  
# 
# #### **Mathematical Definition**  
# For two continuous variables \( X \) and \( Y \):  
# 
# $\text{MI}(X, Y) = \iint p(x, y) \log \left( \frac{p(x, y)}{p(x)p(y)} \right) dx \, dy$  
# where:  
# - $( p(x, y) )$ = joint probability density.  
# - $( p(x), p(y) )$ = marginal densities.  
# 
# **Key Properties**:  
# - $MI (\geq 0)$ (0 means independent).  
# - Higher MI = stronger dependency.  
# 

# ## Used libraries

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


colab = False


# In[3]:


if colab: 
    import sys
    sys.path.append('/content/drive/MyDrive/Colaboracion_Quimica/Main_Codes/AutoEncoders/AmaroX/AmaroX')
    get_ipython().system(' pip install python-telegram-bot')

    from ai_functions import *
    from ai_models import *
    from utilities import *
    from data_manipulation import *
    import pandas as pd
else: 
    from AmaroX.AmaroX.ai_functions import *
    from AmaroX.AmaroX.ai_models import *
    from AmaroX.AmaroX.utilities import *
    from AmaroX.AmaroX.data_manipulation import *
    import pandas as pd
    import sklearn
    import seaborn as sns


# ## Data

# The data presented here corresponds to molecules with their SMILE representation and descriptors, along with the biological activity. Let's first do a quick view of the data shape.

# * All the data presented here was obtained by colaboration with Dr. Erick Padilla at Facultad de Estudios Superiores Zaragoza - UNAM.

# ### Downloading the data

# In[4]:


if colab:
    get_ipython().system(' gdown --id 1cHM9neEhTOZ82UU9HaZkdGdlwE1d4SJT')
    get_ipython().system(' gdown --id 1wZp9pou63ElEYyGGjBeC2pDtscgRgCpj')


# The _data.xlsx_ file contains all the molecular descriptors from the molecule, along with a SMILE representation.

# In[5]:


compounds_md = pd.read_csv("/media/alan-amaro/XicoDisk/Data_Farma_FESZ/Alzheimer/Data/AZH_descriptors_1.csv", low_memory = False)


# In[6]:


compounds_md.head(15)


# In[7]:


y = compounds_md["pChEMBL Value"]
y.shape



# In[9]:


df_x_ = compounds_md.copy()
df_x_ = df_x_.drop('Activity', axis=1) # Este renglón puede ser útil #Opción 2
df_x_ = df_x_.drop('pChEMBL Value', axis=1)
df_x_ = df_x_.drop('Cluster_number', axis=1)
df_x_ = df_x_.select_dtypes("number")  # quitar non_numeric


# In[10]:


df_x_.shape


# In[11]:


x = np.array(df_x_)
x.shape


# ## Applying Mutual Information to Molecular Descriptors

# In[12]:


x_array = np.array(df_x_[ [
'BCUTZ-1l',
 'Xp-4dv',
 'SpMAD_Dzpe',
 'SlogP_VSA10',
 'VR2_A',
 'VSA_EState7',
 'ATSC1i',
 'SssNH',
 'NssCH2',
 'Xpc-5dv',
 'SlogP_VSA8',
 'ATS1d',
 'VE1_A',
 'SssCH2',
 'Xp-7dv',
 'BCUTm-1h',
 'SpDiam_Dzv',
 'piPC6',
 'VSA_EState3',
 'ETA_eta_L',
 'VR1_A',
 'Xp-3dv',
 'nAromAtom',
 'PEOE_VSA2',
 'AMID'
] ])
x_array.shape


# In[13]:


y_array = np.array(y)
y_array.shape


# ## Standarize Features

# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[15]:


x_std = scaler.fit_transform(x_array)
x_std.shape

# ## Splitting Train and Test

# In[17]:


N_BINS=10 ##discretizer, this was 10 before
N_SPLITS=10 ##splitter
TEST_SIZE=2/5 ##splitter


# In[18]:


# dividimos train test con stratified
discretizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="uniform")
splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=N_SPLITS,test_size=TEST_SIZE, random_state=13)
y_discrete = discretizer.fit_transform(np.expand_dims(y_array, axis = -1))
split, split_test = next(splitter.split(np.expand_dims(x_std, axis = -1), y_discrete ))


# In[19]:


x_train = x_std[split]
x_test = x_std[split_test]
y_train = y_array[split]
y_test = y_array[split_test]


# In[20]:


x_train.shape, x_test.shape


# ## K-Folds

# In this section we generate the _sub train_ and validation sets. 

# In[22]:


from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_regression


# In[23]:


# Convert the continuous target into bins
y_binned = np.floor(
    np.interp(y_train, (y_train.min(), y_train.max()), (0, N_BINS ))
    ).astype(int)


# In[24]:


# Create StratifiedKFold with shuffle and seed
skf = StratifiedKFold(n_splits=N_BINS, shuffle=True, random_state=1360)


# In[25]:


# Use the binned labels for stratification
folds = []
for fold, (train_idx, test_idx) in enumerate(skf.split(x_train, y_binned)):
    _x_train, x_val = x_train[train_idx], x_train[test_idx]
    _y_train, y_val = y_train[train_idx], y_train[test_idx]
    folds.append([_x_train, x_val, _y_train, y_val])


# In[26]:


kint = int(input('Selecciona el K: '))


# In[27]:


x_train = folds[kint][0]
x_val = folds[kint][1]
y_train = folds[kint][2]
y_val = folds[kint][3]


# In[28]:


x_train.shape, x_val.shape


# ## Paths

# In[29]:


name = 'DNN_MI_25_FULL_Fold_{}'.format(kint)
if colab:
    folder_path = '/content/drive/MyDrive/Colaboracion_Quimica/Main_Codes/AutoEncoders/models'
else: 
    folder_path = '../models'
    
final_path = os.path.join(folder_path, name)


# ## Callbacks

# In[30]:


callbacks = standard_callbacks(folder_name= name,
                               folder_path= folder_path,
                               patiences= [200, 1000], # 50 epochs without progress, and 2 epochs to reduce LR
                               monitor = 'val_r2_score',
                               flow_direction = 'max')


# ## Seed

# In[31]:


keras.utils.set_random_seed(1360)


# ## DNN Model

# In[32]:


def _DNN():

  inputs = keras.layers.Input((25,))

  _DNN_ = G_Dense(
      inputs = inputs,
      nodes = [ 251, 301, 401, 51, 401 ],
      DP = 5,
      n_final = 1,
      act_func = 'leaky_relu',
      final_act_func = 'relu',
      WI = 'he_normal',
      L1 = 0.0,
      L2 = 0.0,
      use_bias = True
  )

  return keras.models.Model(inputs = inputs, outputs = _DNN_)


# In[33]:


model_DNN= _DNN()


# In[34]:


model_DNN.compile(optimizer = keras.optimizers.Adam(),
                    loss = 'mae',
                    metrics = ['mape', 'r2_score'])


# In[35]:


get_plot_model(model_DNN, folder_path= final_path)


# In[36]:


model_DNN.summary()


# ## Training

# In[37]:


model_trained = model_training(model_DNN,
                               folder_path = final_path,
                               batch_size = 256,
                               num_epochs = 10000,
                               x_train = x_train,
                               y_train = y_train,
                               x_val = x_val,
                               y_val = y_val,
                               callbacks = callbacks)


# ## Save Model

# In[38]:


model_DNN.save( os.path.join(final_path, 'model.h5') )


# ## Evaluate

# **R2 for this Fold**

# In[40]:


loss, accuracy, _ = evaluate_model_regression(model_DNN, x_val, y_val)


# Let's save the predictions to calculate $Q^{2}$ later:

# In[41]:


preds = model_DNN.predict(x_val)


# In[42]:


preds.shape


# In[43]:


y_val.shape


# In[44]:


_preds = np.array([y_val, preds[:,0]])
_preds.shape


# In[46]:


np.save('./preds/preds_{}'.format(kint), _preds)


results = np.array([ evaluate_model_regression(model_DNN, x_val, y_val) ])
np.save('./kfolds/{}'.format(kint), results)

