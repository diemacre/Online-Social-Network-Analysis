"""
Collect data.
"""


import sys
import time
from TwitterAPI import TwitterAPI
import pickle


consumer_key = 'KxKS7dbjBA4ig3PveHJMZKO3y'
consumer_secret = 'JJe9f5jvDCcYcv9fVpcHGQHc4WVQUXXNUG5D4I32Ucodwm7WZ8'
access_token = '303155366-OipyjHaPAbbIzG04ReDLCZjwfxxDqjxdY3qGX3I8'
access_token_secret = '1e4HljCtZD0Piq6g1WrorZ4J8qmVpG2cqyf7ZGf9vb5Ck'



# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    screen_names = []
    with open(filename, "r") as doc:
        for line in doc:
            screen_names.append(line.rstrip('\n'))
    return(screen_names)


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users_info(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO

    users_info = []
    for sname in screen_names:
        request = robust_request(twitter, 'users/lookup', {'screen_name': sname}, max_tries=5)
        user = [i for i in request]
        friends = []
        request = robust_request(twitter, 'friends/ids', {'screen_name': sname, 'count': 5000}, max_tries=5)
        friends = sorted([str(i) for i in request])
        b = {'screen_name': user[0]['screen_name'],'id': str(user[0]['id']),'friends': friends}
        users_info.append(b)
   
    return users_info


def get_tweets(twitter, screen_name, num_tweets):
    """
    Retrieve tweets of the user.

    params:
        twiiter......The TwitterAPI object.
        screen_name..The user to collect tweets from.
        num_tweets...The number of tweets to collect.
    returns:
        A list of strings, one per tweet.
    """

    request = robust_request(twitter, 'search/tweets', {'q': screen_name, 'count': num_tweets})
    tweets = [x['text'] for x in request]

    return tweets


def save_obj(obj, name):
    """
    store, list of dicts
    """
    
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def main():
    print("Import done.")
    twitter = get_twitter()
    print("Get twitter done.")
    screen_names = read_screen_names('users.txt')
    print('Established Twitter connection.')
    print('Read screen names:\n%s' % screen_names)
    users_info = get_users_info(twitter, screen_names)
    save_obj(users_info, 'user_info')
    print("Users info saved.")
    
    tweets = get_tweets(twitter, screen_names[0], 100)
    save_obj(tweets, 'tweets')
    print("%d Tweets saved." % (len(tweets)))


if __name__ == '__main__':
    main()