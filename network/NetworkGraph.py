import instaloader
import networkx as nx
import matplotlib.pyplot as plt

# Initialize Instaloader
L = instaloader.Instaloader()

# Login function
def login_instagram():
    username = input("Enter your Instagram username: ")
    password = input("Enter your Instagram password: ")
    try:
        L.login(username, password)
        print("Login successful!")
    except instaloader.exceptions.BadCredentialsException:
        print("Login failed. Check your username and password.")
        return False
    return True

# Get post owner from URL
def get_post_owner(url):
    shortcode = url.split('/')[-2]
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    return post.owner_username

# Fetch followers and create graph
def create_follower_graph(username):
    profile = instaloader.Profile.from_username(L.context, username)
    G = nx.Graph()
    G.add_node(username)
    
    print(f"Fetching followers of {username}...")
    for follower in profile.get_followers():
        G.add_node(follower.username)
        G.add_edge(username, follower.username)
        
    return G

# Plot the graph
def plot_graph(G, username):
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_size=50, font_size=8)
    plt.title(f"Follower Graph for {username}")
    plt.show()

if __name__ == "__main__":
    if login_instagram():
        urls = input("Enter Instagram post URLs (comma-separated): ").split(',')
        for url in urls:
            url = url.strip()
            #try:
            owner = get_post_owner(url)
            print(f"Post owner: {owner}")
            graph = create_follower_graph(owner)
            plot_graph(graph, owner)
            '''except Exception as e:
                print(f"Error processing URL {url}: {e}")'''