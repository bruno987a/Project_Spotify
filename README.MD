Welcome to our Song Generator Web application!

Its purpose is to help a group of people of up to 10 people to find a playlist that fits everyone’s music taste. 

After downloading the Project folder in the same structure as provided, the connection to the preloaded 
database works effortlessly. Apart from the song ratings no further data must be provided by the user(s). 

To open our Web application on your browser of choice, go to:
https://projectspotify.streamlit.app/

To start the application locally, make sure your terminal is located in the folder that contains
'data', 'app.py', and 'requirements.txt'.
First install all required libraries, by entering the following in your terminal:
pip install -r requirements.txt

Then, start the Streamlit app:
streamlit run app.py

It is clearly divided into the four steps where the user can input his or her preferences. 
First, the number of users has to be determined. Afterwards, the Music Genre, to which type the Playlist is supposed to belong to, can be chosen. Than, a randomized set of 5 songs out of the database coherent with the selected Music Genre will be generated and displayed for rating. After each user has rated every song between 1 and 5 stars, one can click on the “Generate Final Playlist” Button, and a playlist will be created. It will adhere to the length and music genre chosen, and take all ratings of each member into consideration.

You’ve done it! You can now enjoy your newly created playlist with the whole group.

(You'll find our Contribution Matrix and video in the folder additional content)
The files 
