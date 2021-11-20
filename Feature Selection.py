#!/usr/bin/env python
# coding: utf-8

# ![4442f3fa-3429-4a22-bc09-f71be0b207e7.jfif](attachment:4442f3fa-3429-4a22-bc09-f71be0b207e7.jfif)

# <div class="alert alert-block alert-info">
# <h2>Daftar Isi: </h2>
# <h5>Teknik Perhitungan Correlation:</h5>
#     <ol>
#     <li>Pearson</li>
#     <li>Kendall</li>
#     <li>Spearman</li>
#     </ol>
# <h5>Teknik Pemilihan Fitur</h5>
#     <ol>
#     <li>SelectKBest</li>
#     <li>Regresi linier</li>
#     <li>Random Forest</li>
#     <li>XGBoost</li>
#     <li>Penghapusan Fitur Rekursif</li>
#     <li>Boruta</li>    
#     </ol>
# <h5>Metodologi Seleksi Fitur:</h5>
#     <ol>
#     <li>Filter Method</li>
#     <li>Embedded Method</li>
#     <li>Wrapper Method</li>
#     </ol>
# <h5>Metodologi Seleksi Fitur:</h5>
#     <ol>
#     <li>Seleksi Univariat (Univariate Selection)</li>
#     <li>Pentingnya Fitur (Feature Importance)</li>
#     <li>Matriks Korelasi (Correlation Matrix) dengan Heatmap</li>
#     </ol>
# </div>

# # *Seleksi Fitur*
# ---

# Seringkali saat kita memiliki ratusan atau ribuan fitur setelah pembuatan fitur. Terdapat dua masalah :
# 
# 1. Semakin banyak fitur yang kita miliki, semakin besar kemungkinan kita menyesuaikan diri dengan set train dan validasi. Ini akan menyebabkan model bekerja lebih buruk dalam menggeneralisasi ke data baru.
# 2. Semakin banyak fitur yang kita  miliki semakin lama waktu yang dibutuhkan untuk melatih model dan mengoptimalkan hyperparameter.
# 
# Untuk membantu mengatasi masalah sebaiknya menggunakan teknik pemilihan fitur untuk mempertahankan fitur yang paling informatif dalam model kita.
# 
# **Seleksi fitur (feature selection)** adalah proses memilih fitur yang tepat untuk melatih model ML. Untuk melakukan seleksi fitur, kita perlu memahami hubungan antara variables. Hubungan antar dua random variables disebut *correlation* dan dapat dihitung dengan menggunakan *correlation coefficient*.
# 
# Range nilai *correlation coeficient* :
# 
# - Positif maks +1, korelasi positif, artinya kedua variable akan bergerak searah.
# - Negatif maks -1, korelasi negatif, artinya kedua variable akan bergerak berlawanan.
# - Nol, menunjukan antara kedua variable tidak ada correlation.

# ### Teknik Perhitungan Correlation
# ---
# Teknik perhitungan correlation cukup banyak, berikut yang umum digunakan:
# 
# **A. Pearson**
# * Paling umum digunakan.
# * Digunakan untuk numerical data.
# * Tidak bisa digunakan untuk ordinal data.
# * Mengukur linear data dengan asumsi data terdistribusi normal.
# 
# **B. Kendall**
# * Rank correlation measure.
# * Dapat digunakan untuk numerical dan ordinal data, namun tidak untuk nominal data.
# * Tidak diperlukan linear relationship antar variable.
# * Digunakan untuk mengukur kemiripan ranked ordering data.
# * Untuk kondisi normal lebih baik menggunakan Kendall dibandingkan Spearman.
# 
# **C. Spearman**
# * Rank correlation measure
# * Dapat digunakan untuk numerical dan ordinal data, namun tidak untuk nominal data.
# * Tidak diperlukan linear relationship antar variable.
# * Monotonic relationship

# ### Teknik Pemilihan Fitur
# ---
# Teknik pemilihan fitur yang perlu kita ketahui, untuk mendapatkan performa terbaik dari model :
# 
# 1. SelectKBest
# 2. Regresi linier
# 3. Random Forest
# 4. XGBoost
# 5. Penghapusan Fitur Rekursif
# 6. Boruta

# ### Metodologi Seleksi Fitur
# ---
# Ada beberapa metodologi feature selection yang umum digunakan, yaitu:
# 
# **A. Filter Method**
# - Umumnya digunakan pada tahap preprocessing. 
# - Pemilihan features tidak tergantung kepada algoritma ML yang akan digunakan. 
# - Features dipilih berdasarkan score test statistik kolerasi.
# 
# **B. Embedded Method**
# - Feature dipilih saat proses model training. 
# - Menggunakan learning algorithm untuk melakukan variable selection dan feature selection and classification secara simultan.
# - Harus memilih algoritma machine learning yang sesuai.
# 
# **C. Wrapper Method**
# - Menggunakan subset of features untuk melatih model. 
# - Berdasarkan hasil yang dihasilkan dari model sebelumnya, kita tentukan untuk menambah atau membuang features dari subset.
# - Kelemahannya membutuhkan resource besar dalam melakukan komputasi.

# ### Teknik Seleksi Fitur
# ---
# Ada 3 jenis seleksi fitur lainnya dalam slide modul ini, diantaranya:
# 1. **Seleksi Univariat** (Univariate Selection)
# 2. **Pentingnya Fitur** (Feature Importance)
# 3. **Matriks Korelasi** (Correlation Matrix) dengan Heatmap

# ---

# <div class="alert alert-block alert-danger">
# <b>Catatan:</b> Berikut adalah sebagian kecil dari teknik dalam seleksi fitur.
# </div>

# <div class="alert alert-block alert-success">
# <b>Sumber dataset:</b>  https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv
# </div>

# #### Deskripsi variabel dari dataset:
# 
# * battery_power: Total energy a battery can store in one time measured in mAh
# * blue: Has Bluetooth or not
# * clock_speed: the speed at which microprocessor executes instructions
# * dual_sim: Has dual sim support or not
# * fc: Front Camera megapixels
# * four_g: Has 4G or not
# * int_memory: Internal Memory in Gigabytes
# * m_dep: Mobile Depth in cm
# * mobile_wt: Weight of mobile phone
# * n_cores: Number of cores of the processor
# * pc: Primary Camera megapixels
# * px_height: Pixel Resolution Height
# * px_width: Pixel Resolution Width
# * ram: Random Access Memory in MegaBytes
# * sc_h: Screen Height of mobile in cm
# * sc_w: Screen Width of mobile in cm
# * talk_time: the longest time that a single battery charge will last when you are
# * three_g: Has 3G or not
# * touch_screen: Has touch screen or not
# * wifi: Has wifi or not
# * price_range: This is the target variable with a value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).
# 
# 
# 

# <div class="alert alert-block alert-danger">
# <b>Catatan : </b>Jika belum pernah instal library, gunakan perintah berikut secara inline atau melalui terminal.
#     <ol>
#     <li>!pip install pandas</li>
#     <li>!pip install numpy</li>
#     <li>!pip install scikit-learn</li>
#     <li>!pip install matplotlib</li>
#     <li>!pip install seaborn</li>
#     </ol>
# </div>

# ### 1. Seleksi Unvariate
# ---
# Metode paling sederhana dan tercepat didasarkan pada uji statistik univariat. Untuk setiap fitur, ukur seberapa kuat target bergantung pada fitur menggunakan uji statistik seperti  χ2 (chi-square) or ANOVA.
# 
# Uji statistik dapat digunakan untuk memilih fitur-fitur tersebut yang memiliki relasi paling kuat dengan variabel output/target.
# Library scikit-learn menyediakan class *SelectKBest* yang digunakan untuk serangkaian uji statistik berbeda untuk memilih angka spesifik dari fitur. Berikut ini adalah uji statistik chi-square utk fitur non-negatif untuk memilih 10 fitur terbaik dari dataset *Mobile Price Range Prediction*.

# In[1]:


# import library
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[2]:


# memuat data
data = pd.read_csv("train.csv")
data.head()


# In[3]:


# memilih data yang dibutuhkan
X = data.iloc[:,0:20]  #independent colums
y = data.iloc[:,-1]    # target colum i.e price range


# In[4]:


# menerapkan SelectKBest untuk melakukan ekstraksi
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[5]:


# menggabungkan 2 dataframe
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# ### 2. Feature Importance
# ---
# **Feature importance** mengacu pada kelas teknik untuk menetapkan skor ke fitur input ke model prediktif yang menunjukkan *importance* relatif dari setiap fitur saat membuat prediksi. Skor *Feature importance* dapat dihitung untuk masalah yang melibatkan prediksi nilai numerik, yang disebut regresi, dan masalah yang melibatkan prediksi label kelas, yang disebut klasifikasi.
# 
# Skor digunakan dalam berbagai situasi dalam masalah pemodelan prediktif, seperti:
# 
# * Lebih memahami data.
# * Lebih memahami model.
# * Mengurangi jumlah fitur input.
# * memberi  skor untuk setiap fitur data, semakin tinggi skor semakin penting atau relevan fitur tersebut terhadap variabel output
# 
# Inbuilt yang dilengkapi dengan Pengklasifikasi Berbasis Pohon (Tree Based Classifier), kami akan menggunakan Pengklasifikasi Pohon Ekstra untuk mengekstraksi 10 fitur teratas untuk kumpulan data

# In[6]:


# import library
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[7]:


# memuat data
data = pd.read_csv("train.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range


# In[8]:


# melakukan ExtraTreesClassifier untuk mengekstraksi fitur
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[9]:


# melakukan plot dari feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# ### 3. Matriks Korelasi dengan Heatmap
# ---
# 
# * Korelasi menyatakan bagaimana fitur terkait satu sama lain atau variabel target.
# * Korelasi bisa positif (kenaikan satu nilai fitur meningkatkan nilai variabel target) atau negatif (kenaikan satu nilai fitur menurunkan nilai variabel target)
# * Heatmap memudahkan untuk mengidentifikasi fitur mana yang paling terkait dengan variabel target, kami akan memplot peta panas fitur yang berkorelasi menggunakan seaborn library
# 

# In[10]:


# import library
import pandas as pd
import numpy as np
import seaborn as sns


# In[11]:


# memuat data
data = pd.read_csv("train.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range


# In[12]:


# mendapatkan  correlations dari setiap fitur dalam dataset
corrmat = data.corr()
top_corr_features = corrmat.index


# ### Matriks Korelasi dengan Heatmap (lanjutan)
# ---

# In[13]:


# plot heatmap 
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ### Kesimpulan 
# * lihat pada baris terakhir yaitu price range, korelasi antara price range dengan fitur lain dimana ada relasi kuat dengan variabel  ram dan diikuti oleh var battery power ,  px height and px width.
# * sedangkan utk var clock_speed dan n_cores berkorelasi lemah dengan price range
