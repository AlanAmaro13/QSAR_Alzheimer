#!/usr/bin/env python
# coding: utf-8

# # Dense Neural Networks

# Hello there!
# 
# In the previous approach we have considered a linear estimation for the bio-activity. Our result presents an average $R^{2}=0.62$ and a MAPE of $7.19$ In this notebook, we present a new approach by the use of Deep Neural Networks, in this initial case we use only Dense Layers or a Feed Forward. The descriptors used are obtained by the use of Mutual Information (MI). We've first selected those descriptors with a higher mutual than $0.4$, where we've reduced the dimension from 1200 to just 99 descriptors.
# 
# Then, we have selected from the 99 descriptors the one descriptor with the highest MI (piPC4) and have selected two variables that are independent among them. This means, the MI values among them is the lowest value possible.

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
    import seaborn as sns


# In[4]:


import keras_tuner
import sklearn


# ## Data

# The data presented here corresponds to molecules with their SMILE representation and descriptors, along with the biological activity. Let's first do a quick view of the data shape.

# * All the data presented here was obtained by colaboration with Dr. Erick Padilla at Facultad de Estudios Superiores Zaragoza - UNAM.

# In[5]:


compounds_md = pd.read_csv("../Data/AZH_descriptors_1.csv", low_memory = False)


# In[6]:


compounds_md.head(15)


# In[7]:


y = compounds_md["pChEMBL Value"]
y.shape


# In[8]:


# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# First plot
ax1.hist(y, bins = 10, edgecolor = 'black')
ax1.set_title('Hist - pChEMBL Value')
ax1.set_xlabel('Value')
ax1.set_ylabel('# Samples')
ax1.grid()

# Second plot
sns.kdeplot(y, ax = ax2, fill = True)
ax2.set_title('KDE - pChEMBL Value')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.grid()

plt.tight_layout()
plt.show()


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

# In the previous notebook, we have selected 3 molecular descriptors that are independent among them and present a high MI with respect to the bio-activity.

# In[12]:


x_array = np.array(df_x_[ [
    'BCUTZ-1l', 'VR1_A', 'SssNH', 'fragCpx', 'VSA_EState2'
] ])
x_array.shape


# In[13]:


y_array = y
y_array.shape


# ## Standarize Features

# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[15]:


x_std = scaler.fit_transform(x_array)
x_std.shape


# In[16]:


plot_xy([x_std, y_array])


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


# In[21]:


# Crear una figura con dos subplots en horizontal (1 fila, 2 columnas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # figsize ajusta el tamaño

# Graficar la primera curva en el primer subplot
ax1.hist(y_train, color='blue', label='train', bins = 6)
ax1.set_title('Train Density')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# Graficar la segunda curva en el segundo subplot
ax2.hist(y_test, color='red', label='test', bins = 6)
ax2.set_title('Test Density')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Mostrar la figura
plt.show()


# ## Paths

# In[22]:


name = 'DNN_MI_5_5'
if colab:
    folder_path = '/content/drive/MyDrive/Colaboracion_Quimica/Main_Codes/AutoEncoders/models'
else: 
    folder_path = '../models'
    
final_path = os.path.join(folder_path, name)


# ## Callbacks

# In[23]:


callbacks = standard_callbacks(folder_name= name,
                               folder_path= folder_path,
                               patiences= [150, 200], # 50 epochs without progress, and 2 epochs to reduce LR
                               monitor = 'val_r2_score',
                               flow_direction = 'max')


# ## DNN Model

# In[24]:


def _DNN(nodes:list):

  inputs = keras.layers.Input((5,))

  _DNN_ = G_Dense(
      inputs = inputs,
      nodes = nodes,
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


# In[25]:


def compile_model(nodes: list, optimizer, modelo):

  model = modelo(nodes = nodes)

  model.compile(optimizer = optimizer,
                loss = 'mae',
                metrics = ['mape', 'r2_score'])

  return model


# In[26]:


def build_model(hp):

  nodes = [
      hp.Int('Nodes-1', min_value = 1, max_value = 500, step = 1), 
      hp.Int('Nodes-2', min_value = 1, max_value = 500, step = 1),
      hp.Int('Nodes-3', min_value = 1, max_value = 500, step = 1),
      hp.Int('Nodes-4', min_value = 1, max_value = 500, step = 1),
      hp.Int('Nodes-5', min_value = 1, max_value = 500, step = 1),
  ]

  #DP = hp.Int('Dropout', min_value = 0, max_value = 50, step = 2)

  #L1 = hp.Choice('L1', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  #L2 = hp.Choice('L2', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  optimizer = hp.Choice('optimizer', ['adam'])

  if optimizer == 'adam': opt = keras.optimizers.Adam(
        learning_rate = 0.001
    )

  elif optimizer == 'sgd': opt = keras.optimizers.SGD(
        learning_rate = 0.001
    )

  elif optimizer == 'adagrad': opt = keras.optimizers.Adagrad(
        learning_rate = 0.001
    )


  model_f = compile_model(nodes = nodes, optimizer = optimizer, modelo = _DNN)

  return model_f


# In[27]:


build_model(keras_tuner.HyperParameters())


# In[28]:


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective= keras_tuner.Objective('val_r2_score', 'max') ,
    max_trials= 50, # Set to 3
    executions_per_trial = 2,
    overwrite=True,
    directory= final_path,
    project_name="DNN1-MI-KT",
)


# In[29]:


tuner.search_space_summary()


# In[30]:


tuner.search(x_train, y_train, epochs=250, validation_data=(x_test, y_test), batch_size=256)


# In[31]:


file_path = os.path.join(final_path, 'best_models.txt')

with open(file_path, "w") as file:
    # Save the original stdout
    original_stdout = sys.stdout
    try:
        sys.stdout = file  # Redirect stdout to the file
        tuner.results_summary()  # Call your function
    finally:
        sys.stdout = original_stdout


# In[32]:


asyncio.run(send_sms_to_me('HP-5X-5N'))

