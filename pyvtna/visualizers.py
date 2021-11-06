import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualizer:
    def visualize_grid_search(self, t1, t2, t1_norm, t2_norm, trace1, trace2, grid, overlaps,
                              grid_coarse=None, scores_coarse=None):
        pass

    def visualize_opt(self, t1, t2, t1_norm, t2_norm, trace1, trace2, result, trials, scores):
        pass

    @staticmethod
    def plot_traces(data_reader, reaction_names=None, species_names=None, ax=None):
        data, spec_idx = data_reader.select_data(reaction_names, species_names)
        if spec_idx is None:
            species_names = data_reader.species_names
            spec_idx = list(range(len(species_names)))
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for rxn_name, rxn_trace in data.items():
            for i, spec in enumerate(spec_idx):
                ax.scatter(rxn_trace[:, 0], rxn_trace[:, spec + 1], label=f'{rxn_name}: {species_names[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.legend()


class StaticPlotter(Visualizer):
    def __init__(self, fig=None, **fig_kw):
        self.fig = fig
        self.fig_kw = fig_kw

    def reset_fig(self):
        self.fig = None

    def visualize_grid_search(self, t1, t2, t1_norm, t2_norm, trace1, trace2, grid, scores,
                              grid_coarse=None, scores_coarse=None):
        if self.fig:
            a1, a2, a3 = self.fig.subplots(1, 3)
        else:
            self.fig, (a1, a2, a3) = plt.subplots(1, 3, **self.fig_kw)

        best_grid_point = grid[scores.argmax()]
        best_score = scores.max()

        a1.scatter(t1, trace1, label="Rxn 1")
        a1.scatter(t2, trace2, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        a2.plot(grid, scores, c='tab:blue', linewidth=2, label="Cost Function")
        if grid_coarse is not None:
            a2.plot(grid_coarse, scores_coarse, c='tab:orange', linewidth=2, label="Un-Smoothed Cost Function")
        a2.scatter([best_grid_point], best_score, c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Grid')
        a3.scatter(t1_norm, trace1, label='Rxn 1')
        a3.scatter(t2_norm, trace2, label='Rxn 2')
        a3.legend()
        a3.set_title('Best-Fit for Reaction Traces')
        print(f"Best fit achieved for parameter value of {best_grid_point:0.2f}.")
        plt.tight_layout()
        plt.show()

    def visualize_opt(self, t1, t2, t1_norm, t2_norm, trace1, trace2, result, trials, scores):
        if self.fig:
            a1, a2, a3 = self.fig.subplots(1, 3)
        else:
            self.fig, (a1, a2, a3) = plt.subplots(1, 3, **self.fig_kw)
        a1.scatter(t1, trace1, label="Rxn 1")
        a1.scatter(t2, trace2, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        s2 = a2.scatter(trials, scores, c=range(len(trials)), linewidth=2, label="Iterations")
        divider = make_axes_locatable(a2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(s2, cax=cax)
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Grid')
        a3.scatter(t1_norm, trace1, label='Rxn 1')
        a3.scatter(t2_norm, trace2, label='Rxn 2')
        a3.legend()
        a3.set_title('Best-Fit for Reaction Traces')
        print(f"Best fit achieved for an parameter value of {result.x[0]:0.2f}.")
        plt.tight_layout()
        plt.show()

    def visualize_grid_search_single_trace(self, t, t_norm, trace, grid, scores, line,
                                           grid_coarse=None, scores_coarse=None):
        if self.fig:
            a1, a2, a3 = self.fig.subplots(1, 3)
        else:
            self.fig, (a1, a2, a3) = plt.subplots(1, 3, **self.fig_kw)

        best_grid_point = grid[scores.argmax()]
        best_score = scores.max()

        a1.scatter(t, trace, label="Original Rxn")
        a1.legend()
        a1.set_title('Original Product Trace')
        a2.plot(grid, scores, c='tab:blue', linewidth=2, label="Cost Function")
        if grid_coarse is not None:
            a2.plot(grid_coarse, scores_coarse, c='tab:orange', linewidth=2, label="Un-Smoothed Cost Function")
        a2.scatter([best_grid_point], best_score, c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Grid')
        a3.scatter(t_norm, trace, label='Normalized Rxn')
        a3.plot(t_norm, line.predict(t_norm.reshape(-1, 1)), c='tab:orange', label='Linear Fit')
        a3.legend()
        a3.set_title('Best-Fit for Reaction Trace')
        print(f"Best fit achieved for parameter value of {best_grid_point:0.2f}.")
        plt.tight_layout()
        plt.show()

    def visualize_opt_single_trace(self, t, t_norm, trace, result, line, trials, scores):
        if self.fig:
            a1, a2, a3 = self.fig.subplots(1, 3)
        else:
            self.fig, (a1, a2, a3) = plt.subplots(1, 3, **self.fig_kw)
        a1.scatter(t, trace, label="Original Rxn")
        a1.legend()
        a1.set_title('Original Product Trace')
        s2 = a2.scatter(trials, scores, c=range(len(trials)), linewidth=2, label="Iterations")
        divider = make_axes_locatable(a2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(s2, cax=cax)
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Grid')
        a3.scatter(t_norm, trace, label='Normalized Rxn')
        a3.plot(t_norm, line.predict(t_norm.reshape(-1, 1)), c='tab:orange', label='Linear Fit')
        a3.legend()
        a3.set_title('Best-Fit for Reaction Trace')
        print(f"Best fit achieved for an order of {result.x[0]:0.2f}.")
        plt.tight_layout()
        plt.show()
