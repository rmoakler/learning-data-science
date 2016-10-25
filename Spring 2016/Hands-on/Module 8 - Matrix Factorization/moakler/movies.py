import matplotlib.pylab as plt


def movie_search(movies, movie_name):
    return movies.loc[(movies['movie_title'].str.extract('(.*' + movie_name + '.*)').str.len() > 0)][['movie_title', 'genre']]

def movie_plotter(components, movies, movie_id="all", x_buffer=3, y_buffer=2):
    if movie_id == "all":
        plt.scatter(components[:,0], components[:,1])
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
    else:
        x = components[movie_id][0]
        y = components[movie_id][1]

        xs = [x - x_buffer, x + x_buffer]
        ys = [y - y_buffer, y + y_buffer]

        plt.scatter(components[:,0], components[:,1])
        plt.xlim(xs)
        plt.ylim(ys)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        for x, y, title in zip(components[:,0], components[:,1], movies['movie_title']):
            if x >= xs[0] and x <= xs[1] and y >= ys[0] and y <= ys[1]:
                try:
                    plt.text(x, y, title)
                except:
                    pass

