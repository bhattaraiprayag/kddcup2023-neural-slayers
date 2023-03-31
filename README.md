# kddcup2023-neural-slayers
ABOUT THIS PROJECT:
Team project for KDD Cup 2023, as a part of the course IE 672 Data Mining II (FSS2023) at the University of Mannheim.

ABOUT KDD CUP 2023:
✨ Introduction

Modelling customer shopping intentions is crucial for e-commerce stores, as it directly impacts user experience and engagement. Accurately understanding what a customer is searching for, such as whether they are looking for electronics or groceries with the search query “apple”, is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next purchase, has become increasingly popular with the development of data mining and machine learning techniques. However, few studies have explored session-based recommendation under real-world multilingual and imbalanced scenarios.

To address this gap, we present the "Multilingual Shopping Session Dataset," a dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish. The dataset is imbalanced, with fewer French, Italian, and Spanish products than English, German, and Japanese. With this data, we introduce three different tasks:

    predicting the next engaged product for sessions from English, German, and Japanese
    predicting the next engaged product for sessions from French, Italian, and Spanish, where transfer learning techniques are encouraged
    predicting the title for the next engaged product

We hope this dataset and competition will encourage the development of multilingual recommendation systems, which can enhance personalization and understanding of global trends and preferences. This competition aims to provide practical solutions that benefit customers worldwide by promoting diversity and innovation in data science. The dataset will be publicly available to the research community, and standard evaluation metrics will be used to assess model performance.
🗃️ Dataset

The dataset released is anonymized and not representative of the production characteristics.

The Multilingual Shopping Session Dataset is a collection of anonymized customer sessions containing products from six different locales: English, German, Japanese, French, Italian, and Spanish. It consists of two main components: user sessions and product attributes. User sessions are a list of products that a user has engaged with in chronological order, while product attributes include various details like product title, price in local currency, brand, colour, and description.

The dataset has been divided into three splits: train, phase-1 test, and phase-2 test. For Task 1 and Task 2, the proportions for each language are roughly 10:1:1. For Task 3, the number of samples in the phase-1 test and phase-2 test is fixed at 10,000. All three tasks share the same train set, while their test sets have been constructed according to their specific objectives. Task 1 uses English, German, and Japanese data, while Task 2 uses French, Italian, and Spanish data. Participants in Task 2 are encouraged to use transfer learning to improve their system's performance on the test set. For Task 3, the test set includes products that do not appear in the training set, and participants are asked to generate the title of the next product based on the user session.

Table 1 summarizes the dataset statistics, including the number of sessions, interactions, products, and average session length. The dataset will be made publicly available as part of the KDD Cup competition. Each product will be identified by a unique Amazon Standard Identification Number (ASIN), making extracting more information from the web easy. Participants are free to use external sources of information to train their systems, such as public datasets and pre-trained language models, but must declare them when describing their systems beyond the provided dataset.
Language (Locale) 	# Sessions 	# Products (ASINs)
German (DE) 	1111416 	513811
Japanese (JP) 	979119 	389888
English (UK) 	1182181 	494409
Spanish (ES) 	89047 	41341
French (FR) 	117561 	43033
Italian (IT) 	126925 	48788

 

Table 1: Dataset statistics

In addition, we list the column names and their meanings for product attribute data:

    locale: the locale code of the product (e.g., DE)
    id: a unique for the product. Also known as Amazon Standard Item Number (ASIN) (e.g., B07WSY3MG8)
    title: title of the item (e.g., “Japanese Aesthetic Sakura Flowers Vaporwave Soft Grunge Gift T-Shirt”)
    price: price of the item in local currency (e.g., 24.99)
    brand: item brand name (e.g., “Japanese Aesthetic Flowers & Vaporwave Clothing”)
    color: color of the item (e.g., “Black”)
    size: size of the item (e.g., “xxl”)
    model: model of the item (e.g., “iphone 13”)
    material: material of the item (e.g., “cotton”)
    author: author of the item (e.g., “J. K. Rowling”)
    desc: description about a item’s key features and benefits called out via bullet points (e.g., “Solid colors: 100% Cotton; Heather Grey: 90% Cotton, 10% Polyester; All Other Heathers …”)

🕵️‍♀️ Tasks

The main objective of this competition is to build advanced session-based algorithms/models that directly predicts the next engaged product or generates its title text. The three tasks we proposed are:

    Next Product Recommendation
    Next Product Recommendation for Underrepresented Languages/Locales
    Next Product Title Generation

Note that the three tasks share the same training set. However, the objectives of three tasks are different. Details of each tasks are as follows:
Task 1

Task 1 aims to predict the next product that a customer is likely to engage with, given their session data and the attributes of each product. The test set for Task 1 comprises data from English, German, and Japanese locales. Participants are required to create a program that can predict the next product for each session in the test set.

To submit their predictions, participants should provide a single parquet file in which each row corresponds to a session in the test set. For each session, the participant should predict 100 product IDs (ASINs) that are most likely to be engaged, based on historical engagements in the session. The product IDs should be stored in a list and are listed in decreasing order of confidence, with the most confident prediction at index 0 and least confident prediction at index 99.

For example, if product_25 is the most confident prediction for a session, product_100 is the second most confident prediction, and product_199 is the least confident prediction for the same session, the participant's submission should list product_25 first, product_100 next, a lot of other predictions in the middle, and product_199 last.

Input example:
locale 	example_session
UK 	[product_1, product_2, product_3]
DE 	[product_4, product_5]

Output example:
next_item
[product_25, product_100,…, product_199]
[product_333, product_123,…, product_231]

 

The evaluation metric for Task 1 is Mean Reciprocal Rank (MRR).

Mean Reciprocal Rank (MRR) is a metric used in information retrieval and recommendation systems to measure the effectiveness of a model in providing relevant results. MRR is computed with the following two steps: (1) calculate the reciprocal rank. The reciprocal rank is the inverse of the position at which the first relevant item appears in the list of recommendations. If no relevant item is found in the list, the reciprocal rank is considered 0. (2) average of the reciprocal ranks of the first relevant item for each session.

MRR@K=1N∑t∈T1Rank(t),

where Rank(t) is the rank of the ground truth on the top K result ranking list of test session t, and if there is no ground truth on the top K ranking list, then we would set 1Rank(t)=0. MRR values range from 0 to 1, with higher values indicating better performance. A perfect MRR score of 1 means that the model always places the first relevant item at the top of the recommendation list. An MRR score of 0 implies that no relevant items were found in the list of recommendations for any of the queries or users.
Task 2

The goal of this task is similar to Task 1, while the test set is constructed from French, Italian, and Spanish. In task 2, we focus on the performance on these three underrepresented languages. It is encouraged to transfer the knowledge gained from the languages with sufficient data such as English, German, and Japanese to improve the quality of recommendations for French, Italian, and Spanish.

The input/output and evaluation metrics are the same to Task 1.
Task 3

Task 3 requires participants to predict the title of the next product that a customer will engage with, based on their session data. Unlike Tasks 1 and 2, which focus on recommending existing products, predicting new or "cold-start" products presents a unique challenge. The generated titles have the potential to improve various downstream tasks, including cold-start recommendation and navigation. The test set for Task 3 includes data from all six locales, and participants should submit a single parquet file containing the generated titles for each row/session in the input file. The title should be saved in string format.

Input example:
locale 	example_session
UK 	[product_1, product_2, product_3]
DE 	[product_4, product_5]

Output example:
next_item_title
"toilet paper tube"
"bottle of ink"

The evaluation metrics for this task is bilingual evaluation understudy (BLEU). BLEU is a metric used to evaluate the quality of natural language generation, by comparing generation candidate to one or more references. BLEU is computed using a couple of ngram modified precisions. Specifically,

BLEU=BP⋅exp⁡(∑n=1Nwnlog⁡pn)

where BP is the brevity penalty. N is the maximum n-gram length used for calculating precision scores. wn is the weight assigned to each n-gram precision score. exp is the exponential function. pn is the precision score for each n-gram. The precision score pn is the ratio of the number of n-grams in the candidate that appear in any of the reference, to the total number of n-grams in the candidate. Mathematically, pn is calculated as follows:

pn=∑s∈Cmin(countnC(s),maxr∈Rcountnr(s))∑s∈CcountnC(s)

where countnC(s) is the count of n-gram s in the candidate C. countnr(s) is the count of n-gram s in the reference r∈R, where R are a set of references. The brevity penalty (BP) is a correction factor that penalizes the generation candidate that is too short compared to the reference. The brevity penalty is calculated as follows:

BP={1,if Lc>Lrexp⁡(1−LrLc),ifLc≤Lr

where Lc is the length of the generation and Lr is the length of the shortest reference. In general, the BLEU score ranges from 0 to 1, with higher scores indicating better generation.

We set N=4 (i.e., BLEU-4) with wn=1/N for this task.

Leaderboard & Evaluations
Each task will have its separate leaderboard, which will be maintained throughout the competition for models evaluated on the public test set. At the end of the competition, a private leaderboard will be maintained for models evaluated on the private test set. This latter leaderboard will be used to decide the winners for each task in the competition. The leaderboard on the public test set is meant to guide participants on their model performance and compare it with other participants.
