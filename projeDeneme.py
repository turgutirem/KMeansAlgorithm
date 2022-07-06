#Gerekli kutuphaneleri import ettik
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage


#datayı df içerisne atarak okuyoruz
df = pd.read_csv('C:/Users/irem turgut/Desktop/veri seti/CC GENERAL.csv')
# yapılacak işlem zamnında orjinal datanın kaybolmaması için "data" kopyalıyourz
data = df.copy()
# sütunların isimleri büyük harf olduğu için küçük harfe dönüştürüyoruz
data.columns = data.columns.str.lower()
# ilk 5 satıra bakarak data içeriğini gözlemlemeye çalışıyoruz
data.head()
# datanın uç noktalarını ve ayrıntılı özellikleri için describe uyguluyoruz
data.describe()
# satır ve sütun sayısına bakıyoruz
data.shape
# data hakkında sütun isimlerini ve içerik sayısına bakarak hangi türde veri olduğuna ve kabaca eksik veri var mı ona bakıyoruz
data.info()
# null yani boş değer sorguluoruz (false = 0) ve topluyoruz bu şekilde kayıp veriyi tespit edebiliyoruz
data.isnull().sum().sort_values(ascending=False)
# minumum_payments verisinden 313 veri eksik olduğunu
# credit_limit değerinden 1 veri eksik olduğunu anlıyoruz
# eksik olan verilerin yüzdesini hesaplayarak eksik olan verinin silinmesi durumunda ne kadar önem sahip olduğunu görebiliriz
(df.isnull().sum()/data['cust_id'].count())*100
data.dropna(inplace=True)
# Cust_ID, model oluşturma için ihtiyaç duymayacağımız bir sütundur, bu yüzden çıkarıyoruz
data.drop('cust_id', axis=1, inplace=True)
print(data.shape)
o_cols = data.select_dtypes(include=['object']).columns.tolist()
num_cols = data.select_dtypes(exclude=['object']).columns.tolist()
# her sütunun dağılımını göreselleştiriyoruz
data[num_cols].hist(bins=15, figsize=(20, 15), layout=(5, 4))
sns.pairplot(data)
plt.show()
plt.rcParams['figure.figsize'] = (15,7)

sns.countplot(y=data['balance_frequency'],order = data['balance_frequency'].value_counts().index)
plt.ylabel('Bakiye Frekans Puanı (0-1)')
plt.title('Bakiye Sayısı Sıklık Skoru', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(data['purchases'], color='orange', bins=150)
plt.title('Satın Almaların Dağılımı', size=20)
plt.xlabel('Satın Alımlar')
plt.show()

plt.subplot(1,2,1)
sns.distplot(data['oneoff_purchases'],color='green')
plt.title('Tek Seferlik Satın Alma Dağıtımı', fontsize = 20)
plt.xlabel('Miktar')


plt.subplot(1,2,2)
sns.distplot(data['installments_purchases'], color='red')
plt.title('Taksitli Satın Alma Dağıtımı', fontsize = 20)
plt.xlabel('Miktar')
plt.show()
plt.rcParams['figure.figsize'] = (12,15)

plt.subplot(2,2,1)
sns.scatterplot(data['purchases'],data['credit_limit'])
plt.title('Kredi Limiti ve Alımlar', fontsize =20)
plt.xlabel('Satın Alımlar')
plt.ylabel('Kredi Limiti')

plt.subplot(2,2,2)
sns.scatterplot(data['balance'],data['credit_limit'])
plt.title('Credit Limit And Balance', fontsize =20)
plt.xlabel('Balance')
plt.ylabel('Kredi Limiti')

plt.subplot(2,2,3)
sns.scatterplot(data['oneoff_purchases'],data['credit_limit'])
plt.title('Kredi Limiti ve Tek Seferlik Alımlar', fontsize =20)
plt.xlabel('Tek seferlik satın almalar')
plt.ylabel('Kredi Limiti')

plt.subplot(2,2,4)
sns.scatterplot(data['installments_purchases'],data['credit_limit'])
plt.title('Kredi Limiti Ve Taksitli Alımlar', fontsize =20)
plt.xlabel('Taksitli Alımlar')
plt.ylabel('Kredi Limiti')

plt.show()
plt.rcParams['figure.figsize'] = (15,20)

plt.subplot(3,2,1)
sns.violinplot(data['balance_frequency'])
plt.title('Denge Frekansı', fontsize =20)
plt.xlabel('Denge Frekansı')

plt.subplot(3,2,2)
sns.violinplot(data['purchases_frequency'])
plt.title('Satın Alma Sıklığı', fontsize =20)
plt.xlabel('Satın Alma Sıklığı')

plt.subplot(3,2,3)
sns.violinplot(data['oneoff_purchases_frequency'])
plt.title('tek seferlik satın alma Sıklığı', fontsize =20)
plt.xlabel('tek seferlik satın alma Sıklığı')

plt.subplot(3,2,4)
sns.violinplot(data['purchases_installments_frequency'])
plt.title('Taksit Alım Sıklığı', fontsize =20)
plt.xlabel('Taksit Alım Sıklığı')

plt.subplot(3,2,5)
sns.violinplot(data['cash_advance_frequency'])
plt.title('Nakit Avans Sıklığı', fontsize =20)
plt.xlabel('Nakit Avans Sıklığı')
plt.show()

plt.rcParams['figure.figsize'] = (15,8)
plt.subplot(1,2,1)
sns.boxenplot(data['purchases_trx'])
plt.title('Toplam Satın Alma Sayısı')
plt.xlabel('Satın Alımlar')

plt.subplot(1,2,2)
sns.boxenplot(data['cash_advance_trx'])
plt.title('Nakit Avans İle Yapılan Toplam İşlem Sayısı')
plt.xlabel('Satın Alımlar')
plt.show()
plt.figure(figsize=(20,20))
corr_data = data.corr()
sns.heatmap(corr_data,annot=True)
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled.shape
hier_cluster = linkage(data_scaled, method='ward')
plt.figure(figsize=(10,9))
plt.title('Hiyerarşik Kümeleme Dendrogram')
plt.xlabel('Gözlemler')
plt.ylabel('Mesafe')
dendrogram(hier_cluster, truncate_mode='level', p = 5, show_leaf_counts=False, no_labels=True)
plt.show()

wcss= []

for i in range(1,11): 
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(data_scaled)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11),wcss, marker='o', linestyle='--')
plt.title('Optimal Kümeleri Bulmak İçin Dirsek Yöntemi', fontsize =20)
plt.xlabel('Kümeler')
plt.ylabel('wcss')

plt.show()

# modeli 4 küme olarak ayarlayıp o şekilde tekrardan algoritmayı uyguluyoruz
km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km.fit_predict(data_scaled)

data['label'] = label
data['constant'] = 'constant'
plt.rcParams['figure.figsize'] =(25,40)

for num in range(0,17):
    ax = plt.subplot(5,4,num+1)
    col = data.columns[num]
    sns.stripplot(data['constant'],data[col], ax=ax, hue=data['label'])
    plt.xlabel('constant')

plt.show()
pca = PCA(n_components = 7)  
pca.fit(data_scaled)
data_scaled.shape
# x datasını 2 boyuta döüştürüyorum
x_pca = pca.transform(data_scaled)
x_pca.shape
#gerçek datayı ne kadar temsil ettiğine bakıyoruz
print("variance ratio: ", pca.explained_variance_ratio_)
#datamın ne kadaraını kaybettiğimizi öğreniyoruz 
print("sum: ",sum(pca.explained_variance_ratio_))
#data["p1"] = x_pca[:,0]
#data["p2"] = x_pca[:,1]

x = x_pca[:,0]
y = x_pca[:,1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow'}
  
data = pd.DataFrame({'x': x, 'y':y, 'label':data['label']}) 
groups = data.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("Customer Segmentation based on Credit Card usage")
plt.show()
plt.figure(figsize=(20,20))
corr_pca = data.corr()
sns.heatmap(corr_pca,annot=True)
plt.show()