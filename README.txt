Readme
———————

I have written three codes, one for user-based collaborative filtering, second for item-based collaborative filtering and the third for hybrid-based collaborative filtering.

First, move to the folder and copy the files ratings.csv, toBeRated.csv, users.csv, and movies.csv to the downloaded “Code” folder. Now, enter the commands explained as follows:

1. User-Based Collaborative Filtering:
python userBased.py ratings.csv toBeRated.csv cosine jaccard pearson

After the above command finish executing, it will provide result1.csv as the output file which will have the predicted ratings. Also, rmse_user.txt will be an output which shows the rmse obtained in all the three types of similarity and also shows which one is the best.

2. Item-Based Collaborative Filtering:
python itemBased.py ratings.csv toBeRated.csv cosine jaccard pearson

After the above command finish executing, it will provide result2.csv as the output file which will have the predicted ratings. Also, rmse_item.txt will be an output which shows the rmse obtained in all the three types of similarity and also shows which one is the best.

3. Hybrid-Based Collaborative Filtering:
python hybrid.py ratings.csv toBeRated.csv users.csv movies.csv cosine jaccard pearson

After the above command finish executing, it will provide result3.csv as the output file which will have the predicted ratings. Also, rmse_hybrid.txt will be an output which shows the rmse obtained in all the three types of similarity and also shows which one is the best.
