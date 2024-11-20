import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class ObjIter:
    """
    Describes the objective value in an iteration
    """

    def __init__(self, objective, iteration):
        self.objective = objective
        self.iteration = iteration

    def __str__(self):
        return "Iteration: " + str(self.iteration) + ", Obj: " + str(self.objective)


class Stats:
    # Starts with the initial objective value at iteration 0.
    obj_over_time: [ObjIter]

    def __init__(self, title: str, start_time: float, end_time: float, iterations: int, final_objective: int,
                 obj_over_time: [ObjIter]):
        self.title = title
        self.start_time = start_time
        self.end_time = end_time
        self.iterations = iterations
        self.final_objective = final_objective
        self.obj_over_time = obj_over_time

    def get_run_time(self):
        return self.end_time - self.start_time

    def get_iterations(self):
        return self.iterations

    def get_final_objective(self):
        return self.final_objective

    def get_obj_over_time(self):
        return self.obj_over_time

    def print_stats(self):
        print("===== Stats =====")
        print("Runtime: " + str(self.get_run_time()))
        print("Iterations: " + str(self.iterations))
        print("Final objective: " + str(self.final_objective))

    def show_plot(self, title):
        x_points = list(range(len(self.obj_over_time)))
        y_points = []
        for i in range(len(self.obj_over_time)):
            y_points.append(self.obj_over_time[i].objective)

        plt.plot(x_points, y_points, 'o')

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        plt.title(self.title + ", " + title)
        description = "Runtime: " + f"{self.get_run_time():.5f}s"
        if self.get_iterations() > 0:
            description += "\n" + "Iterations: " + str(
                self.iterations)
        description += "\n" + "Final objective: " + str(self.final_objective)
        plt.text(0.95, 0.95, description,
                 fontsize=10,
                 ha='right',
                 va='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        plt.show()


class MultiStats:
    def __init__(self, stats: [Stats]):
        self.stats = stats

    def plot_stats(self, title: str):
        colors = ["blue", "green", "red", "yellow", "black", "cyan"]
        c = 0

        for stat in self.stats:
            x_points = list(range(len(stat.obj_over_time)))
            y_points = []
            for i in range(len(stat.obj_over_time)):
                y_points.append(stat.obj_over_time[i].objective)

            plt.plot(x_points, y_points, label=stat.title + f", {stat.get_run_time():.5f}s", color=colors[c],
                     marker='o')

            c += 1

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        plt.title("Comparison: " + title)
        plt.legend()
        plt.show()
