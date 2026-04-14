from flask import Flask, render_template, request, redirect, url_for, flash  # importing Flask and related functions
import pandas as pd # importing pandas to handle csv file data
import pickle # importing pickle to load saved models

# creating the Flask app
app = Flask(__name__)

# setting the secret key for session security
app.secret_key = 'supersecretkey'

# loading the data from CSV and taking unique clothes (total 32 clothes listed)
data = pd.read_csv('assignment3_II.csv')
unique_clothes = data[['Clothes Title', 'Clothes Description']].drop_duplicates()

# loading the vectorizer and model files from assignment 3 milestone I
with open('models/count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/logistic_regression_count.pkl', 'rb') as f:
    model = pickle.load(f)

new_reviews = {}  # creating a dictionary to store new reviews

# getting the recommended products from the data
# however it is recommending all 32 unique products under each item, hence (limitation of my code)
def get_recommended_products():
    """Retrieve items with positive recommendations."""
    recommended = data[data['Recommended IND'] == 1]
    return recommended[['Clothes Title', 'Clothes Description']].drop_duplicates()

# defining the home route
@app.route('/')
def home():
    return render_template('home.html')  # rendering the home page

# defining the browse route
@app.route('/browse', methods=['GET', 'POST'])
def browse():
    search_results = None
    if request.method == 'POST': # checking if the search form is submitted
        search_term = request.form.get('search_term', '').lower()
        matching_titles = unique_clothes[
            unique_clothes['Clothes Title'].str.contains(search_term, case=False)
        ]
        search_results = matching_titles  # storing matching items

    return render_template('browse.html', results=search_results)

# defining the item route to show product details and reviews
@app.route('/item/<title>', methods=['GET', 'POST'])
def item(title):
    item_data = unique_clothes[unique_clothes['Clothes Title'] == title].iloc[0]
    item_title = item_data['Clothes Title']
    item_description = item_data['Clothes Description']
    item_reviews = data[data['Clothes Title'] == title]

    # getting new reviews specific to this product
    product_reviews = new_reviews.get(title, [])

     # handling new review submission
    if request.method == 'POST':
        review_title = request.form.get('review_title')
        review_description = request.form.get('review_description')
        rating = int(request.form.get('rating'))

        # predicting the recommendation using the model
        review_text = f"{review_title} {review_description}"
        review_vector = vectorizer.transform([review_text])
        recommendation = model.predict(review_vector)[0]
        model_rec = 'Yes' if recommendation == 1 else 'No'

        # storing the new review under the correct product title
        new_review = {
            'Title': review_title,
            'Review Text': review_description,
            'Rating': rating,
            'Model Recommendation': model_rec,
            'Customer Feedback': None  # to be updated later once customer overrides the model's recommendation to yes/no
        }
        if title not in new_reviews:
            new_reviews[title] = []  # initializing if no reviews exist yet
        new_reviews[title].append(new_review)

        flash('Your review is now live! Thank you for sharing your thoughts.')
        return redirect(url_for('item', title=title))

    # getting recommended products to display on the product page
    recommended_products = get_recommended_products()

    return render_template(
        'item.html',
        title=item_title,
        description=item_description,
        reviews=item_reviews,
        new_reviews=product_reviews,
        recommended_products=recommended_products
    )

# defining the route to edit customer feedback on reviews
@app.route('/edit_review', methods=['POST'])
def edit_review():

    # getting the review title, customer feedback, and model's recommendation from the form
    review_title = request.form.get('review_title')
    customer_feedback = request.form.get('recommendation')
    model_rec = request.form.get('model_recommendation')

    # updating the customer's feedback
    for reviews in new_reviews.values():
        for review in reviews:
            if review['Title'] == review_title:
                review['Customer Feedback'] = customer_feedback
                break # exiting the loop once the review is updated

    # not working (limitation of my code)
    flash(f'Thank you for your feedback: {customer_feedback}.')
    
    # redirecting back to the previous page
    return redirect(request.referrer)

# running the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)

'''
-----------------------------------------------------------------------------------------------------------------

# =======================
# Additional Information
# =======================

# All references used in this assignment are listed in the README.txt file.
#
# I used the following code in Assignment 3 - Milestone 1 to create the vectorizer
# and the logistic regression model saved as .pkl files. I chose to go with 
# CountVectorizer because it is simple and easy to use. It converts text into 
# numbers by counting the frequency of words, which works well for my dataset.
#
# I did not use other vectorizers like TF-IDF because I wanted to keep things 
# simple and focus on building a working model first. TF-IDF focuses more on 
# rare words, but in my case, all reviews and titles are already well-balanced, 
# so I didn’t feel the need for it. 
#
# The logistic regression model I trained gave me an accuracy of 89.34%. I am 
# happy with this score because it shows that the model is doing a good job in 
# predicting whether the product will be recommended or not.
#
# Below is the code I used to generate the vectorizer and logistic regression model:

# importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# combining the 'Final Cleaned Review' and 'Final Cleaned Title'
data_df['Combined Feature'] = data_df['Final Cleaned Title'] + " " + data_df['Final Cleaned Review']

# extracting target labels (Recommended IND)
X = data_df['Combined Feature']
y = data_df['Recommended IND']

# initializing CountVectorizer and transform the combined text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# training a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# evaluating the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# saving the vectorizer and model as .pkl files
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('logistic_regression_count.pkl', 'wb') as f:
    pickle.dump(model, f)

print("CountVectorizer and Logistic Regression model saved as .pkl files.")

-----------------------------------------------------------------------------------------------------------------
'''