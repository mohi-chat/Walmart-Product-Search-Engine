# Walmart-Product-Search-Engine

### Project Demo video: https://www.youtube.com/watch?v=t9r7HDujc7g
A search engine to look up products listed in Walmart.com. Developed as part of TAMU Datathon 2020.

Inspired by the search engine challenge and wanting to try our hand at a real-world dataset and the Walmart challenge gave us one of the best opportunities.

Task I -: This was one of the most laborious and frustrating tasks as we faced a lot of forbidden issues and captcha but this was a great learning step for us as we came across a lot of workarounds and things that we can face while scrapping data and processing it. We used a selenium based crawler with a headless server (phantomjs) for crawling. BeautifulSoup library was used to post process the raw html response. We used a random surfer model to hop from product to product to build up our product database. The final data we crawled can be found here: https://github.com/mohi-chat/Walmart-Product-Search-Engine/blob/main/FlaskApp/search-app/combined.csv

Task II -: A search engine to look up products listed on Walmart.com. Developed as part of the TAMU Datathon 2020. We started by trying different searching models such as boolean retrieval and used different ranking methods such as vector space model, BM25, TF-IDF scores, etc. We finally ended up using BM25 for ranking and boolean retrieval for matching relevant products. The reason we chose Boolean Retrieval is that the data we are working on is an eCommerce data and while searching for products we usually want exact term matches that we have in our query and Boolean Retrieval does exactly that for us and in a quick manner.

Task III -: Initially we tried with meanshift clustering, but we didn't get good results with meanshift clustering algorithm. We tried OPTICS as well. In the end, we felt hierarchical clustering would make sense because of the structure of the data. We implemented Agglomerative hierarchical clustering using manhattan distance as the metric, with a distance threshold as 20. Clustering was done on Word2Vec Embeddings with 100-word dimensions.

Task IV -: Finally we developed a web page using flask API, we created a simple UI due to lack of time. Finally, we used the clustering methods when our boolean retrieval method showed low confidence score. After much analysis, we cam across a threshold of 0.35 for the best result.
