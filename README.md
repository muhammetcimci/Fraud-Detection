# Fraud-Detection

This  project was handled to finish the Data Science Bootcamp training organized by the VBO organization. In this project we worked as a group, and it took one month. The group members are:
- Berkan ACAR
- Mert Ozan INAL
- Muhammed CIMCI
- Ismail KAYA
- Umit CEYLAN


# Scope of The Project

The aim of the project is benchmarking machine learning models on a challenging large-scale dataset to determine if transaction is fraud or not. 

# About Data Set
There was a competition hosted by IEEE Computational Intelligence Society [(IEEE-CIS)](https://cis.ieee.org/) on [Kaggle](https://www.kaggle.com/) in 2019. The data is originally coming from the world’s leading payment service company, [Vesta Corporation](https://trustvesta.com/). You can read about the fraud detection competition and find the data set [here.](https://www.kaggle.com/c/ieee-fraud-detection). 


 #  Abstract

The data set used in this project was given as 4 different csv files, two of which are Train sets and the other two belong to the Test set. I fixed the structural errors in the column names of the files and combined the files with left join with Train and Test set. Since more than 95% of the variables given in terms of data privacy are kept confidential, I grouped the variables with different pattern determination techniques. After applying the EDA analysis to the grouped variables, I defined the userid to identify each person to whom the transactions belong. Based on Userid, I did feature engineering and got a Train set with dimension of 590540 x 284  before I entered it into the model. I predicted the Test set using tree-based models like XGBoost, LightGBM and CatBoost using this Train set I obtained. I used the GridSearch method to determine the hyperparameters of these models. Then I applied Kfold to the models and achieved higher accuracy results. The accuracy of these 3 models turned out to be very close to each other. Accurcy values of XGBoost, LightGBM and CatBosst were 92%, 91% and 90%, respectively.

-----------------------------------------------------------------


1. Bu çalışma, Veri Bilimi Okulu tarafından organize edilen Data Science Bootcamp eğitimini bitirme projesi olarak; Muhammed Çakmak, İsmail Kaya, Muhammed Cimci, Ümit Ceylan, Berkan Acar ve Mert Ozan İnal' ın içinde bulunduğu 6 kişilik bir ekip tarafından yapıldı. 

Çalışmanın verileri 2019 yılı september ve october ayları arasında Kaggle platformunda IEEE-CIS organizasyonu tarafından düzenlenen Fraud Detection yarışmasından alındı. 

Modelin başarısı yarışmanın başarı metriğine göre, öngörülen olasılık ile gözlenen hedef arasındaki ROC eğrisi altında kalan alan üzerinden değerlendirildi.

Muhammed Çakmak, İsmail Kaya, Muhammed Cimci, Ümit Ceylan, Berkan Acar ve Mert Ozan İnal' ın katkılarıyla yaptığımız çalışmalarımızın özetini aşağıda açıklamış olacağız. 

Amacımız müşteri işlemlerinin sahtekarlık olup olmadığını tahmin etmek. (detaylandırılabilir)

-------------

Using the data, I analyzed factors that correlated with loans being repaid on time, and did some exploratory visualization and analysis.  I then created a model that predicts the chance that a loan will be repaid given the data surfaced on the LendingClub site.  This model could be useful for potential lenders trying to decide if they should fund a loan.  You can see the exploratory data analysis in the `Exploration.ipynb` notebook above.  You can see the model code and explanations in the `algo` folder.


Verileri kullanarak, zamanında geri ödenen kredilerle ilişkili faktörleri analiz ettim ve bazı keşifsel görselleştirme ve analizler yaptım. Daha sonra LendingClub sitesinde ortaya çıkan veriler göz önüne alındığında bir kredinin geri ödenme şansını tahmin eden bir model oluşturdum. Bu model, bir krediyi finanse edip etmemeye karar vermeye çalışan potansiyel kredi verenler için yararlı olabilir. Keşifsel veri analizini yukarıdaki `Exploration.ipynb` not defterinde görebilirsiniz. Model kodunu ve açıklamaları `algo` klasöründe görebilirsiniz.

- eksik verileri -1 ile neden doldurduk, datanın hepsini pozitife çevirme politikasının nedeni.

- encoding yöntemleri (factorize kullanıldı)

- frequence encoding

- feature engineering 


------------
Some bullet points with interesting observations you found in exploration

- veriler birbirlerine transactionID değişkeni ile bağlanabilen iki tabloda sunulmuştur. veri incelendiğinde;
- verinini 6 aylık zaman serisi olduğu, nan değerler görselleştirildiğinde, aynı kaynaktan gelen veya farklı nedenlerden dolayı bazı değişkenlerde nan paterni yakaladık. oluşan pattern gruplarının kendi aralarındaki korelasyonlarına göre boyut azaltma işlemleri yapıldı. burada dikkatimizi tuhaf olarak v sütunlarının yanında bazı d ve id sütunlarının da bu patternlerde elendiğini gördük. bunu aynı bilgiyi taşıyan sütunları elemek (çoklu doğrusal bağlantı problemini engellemek). kategorik değişkenlerden user id üretip bunlar üzerinden nümerik değişkenlerin aggregationları yapıldı. transectiondt gün ay ve yıla dünuştürülerek kullanıldı. ... işlemleri yapıldı.


daha sonra user idler ... nedenlerden dolayı drop edildi.

-----

Any interesting charts or diagrams you created

-----

Information about the model, such as algorithm

•	Hangi modeller kullanılacağını belirlemek.
•	En uygun model belirleyerek parametre ayarlamalarını yapmak, modeli geliştirmek.
•	Gerekli ise veri manipülasyonu ve tekrar modelin kurulması
- 3 model xgboost,lightgbm ve catboost modellerini kullanarak grid search yöntemi kullanarak hiperparametre tuning yaptık.
- her üç model için kfold kullanarak birer tahmin yaptık ayrıca modellerin sade halleriyle tahmin yaptık ve bunların başarısını karşılaştırdık.
- 6 modelin sonuçları tablo ile gösterilecek.
- catboost modelinde kategorik featurelara duyarlılığını test ettik. kategorik feature özelliği tercih edildiğinde sonucun her deneme yüksek başarı g
göstermesinden dolayı kategorik feature özelliği kullanıldı.
- başarı metrikleri denendi.

------

Error rates and other information about the predictions

- dengesiz veri örneği olduğu için doğru başarı metriğinin kullanılması gerekliydi. burada önemli olan yani hedef fraud işlemlerinin doğru tespit edilmesi. yani tp oranının artırılması.

-----

Any notes about real-world usage of the model

- bankacılık sektörü için kullanılır.


-------


Final modeli oluşturmak için modeli eğitirken, eğitim ve test verilerinin toplam hacmi 5 GB’ ın üstündeydi.
Tabloları birleştirip veriyi manipüle etmeye başladığımızda eğitim verimizde; 590.540 gözlem – 434 değişken,
test verimizde ise 506.691 gözlem – 433 değişken vardı.

# •	Dosya hacmini küçülttük. sonucta ... kadar gözlem, ... değişken oldu ve dosya boyutu memory reduce yapılarak ... gb boyutuna indirildi.






