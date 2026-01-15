# +
import numpy as np
import array
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
import io
from abc import ABC, abstractmethod
from tabulate import tabulate
from typing import Callable, Sequence, Optional, Tuple
from copy import copy, deepcopy
from matplotlib import colors

from IPython.display import display, Markdown, Latex


# -
@dataclass
class Dataset:
    x:  np.array
    dx: np.array
    y:  np.array
    dy: np.array

@dataclass
class GraphingOptions:
    x_label: str = ''
    y_label: str = ''
    x_units: str = ''
    y_units: str = ''
    
    data_marker:      str   = '.'
    data_marker_size: int   = 2
    data_linestyle:   str   = ''
    data_alpha:       float = 0.80
    data_color:       str   = 'C0'
    
    model_marker:     str   = ''
    model_linestyle:  str   = '-'
    model_linewidth:  int   = 2
    model_alpha:      float = 1.0
    model_color:      str   = 'darkred' 
    
    data_round: int = 1

    def set_labels(self, xlabel=None, ylabel=None):
        if xlabel is None:
            plt.xlabel(f"{self.x_label} ({self.x_units})")
        else:
            plt.xlabel(xlabel)

        if ylabel is None:
            plt.ylabel(f"{self.y_label} ({self.y_units})")
        else:
            plt.ylabel(ylabel)
            
    def plot_data(self, x, y, dx, dy, label=None, color=None):
        base_color = color if color is not None else self.data_color
        base_rgb = colors.to_rgb(base_color)

        plt.errorbar(
            x, y,
            xerr=dx,
            yerr=dy,
            marker=self.data_marker,
            markersize=self.data_marker_size,
            linestyle=self.data_linestyle,

            # opaque markers
            color=base_color,

            # translucent error bars
            ecolor=(*base_rgb, 0.25),

            elinewidth=1.0,
            capsize=3,
            label=label,
        )
    
    def plot_model(self, model_x, model_y):
        plt.plot(model_x, model_y, 
                 marker    = self.model_marker, 
                 linestyle = self.model_linestyle, 
                 linewidth = self.model_linewidth,
                 alpha     = self.model_alpha,
                 color     = self.model_color,
                 label     = f'Fit')
    
    def plot_residuals(self, x, residuals, y_uncert):
        plt.title("Residuals")
        plt.ylabel(f"Residual y-y_fit [{self.y_units}]")
        plt.errorbar(x, residuals, yerr=y_uncert, 
                     marker     = self.data_marker,
                     markersize = self.data_marker_size,
                     linestyle  = self.data_linestyle,
                     alpha      = self.data_alpha,
                     color      = self.data_color,
                     label      = "Residuals")
        plt.axhline(y=0, 
                    marker    = self.model_marker, 
                    linestyle = self.model_linestyle, 
                    linewidth = self.model_linewidth,
                    alpha     = self.model_alpha,
                    color     = self.model_color, 
                    label     = f'Fit')

    @staticmethod
    def save_graph_and_close():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = Image.open(buf)
        plt.close()
        return graph
        
    def default_title(self):
        return f'{self.y_label} vs. {self.x_label}, Round {self.data_round}'

@dataclass
class Model:
    """Base class for fit models."""
    
    fit_function:  Callable = None
    param_names:   Sequence[str] = field(default_factory=list)
    param_values:  np.ndarray = field(default_factory=lambda: np.array([]))
    param_uncerts: np.ndarray = field(default_factory=lambda: np.array([]))
    param_bounds:  Optional[Tuple[np.ndarray, np.ndarray]] = None

    def values(self):
        return tuple(self.param_values)

    def uncertainties(self):
        return tuple(self.param_uncerts)

    def labels(self):
        return tuple(self.param_names)
    
    def has_bounds(self) -> bool:
        return self.param_bounds is not None

    def bounds(self):
        if self.param_bounds is None:
            raise ValueError("Model has no parameter bounds")
        return self.param_bounds

    def update_fit_results(self, fit_params, fit_errors):
        self.param_values = np.array(fit_params)
        self.param_uncerts = np.array(fit_errors)
    
    def tabulate(self, units=None):
        def apply_units(label, i):
            if units is None or len(units) <= i:
                return label
            else:
                return f'{label} ({units[i]})'
        
        header = [ 'Measurement', 'Value', 'Uncertainty' ]
        rows = [ 
            [ apply_units(label, i), '%.3e' % value, '%.3e' % uncert ] \
                for i, (label, value, uncert) in \
                    enumerate(zip(self.labels(), 
                                  self.values(), 
                                  self.uncertainties())) 
        ]
        
        return tabulate(rows, header, tablefmt='grid')

# **Exponential Fit**
#
# $$f(x) = A \exp(-x/\tau) + C$$

class ExponentialOffsetModel(Model):
    def __init__(self, amplitude, time_constant, offset):
        def fit_function(x, amplitude, time_constant, offset):
            return amplitude * np.exp(-x/time_constant) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Offset'],
            param_values=np.array([amplitude, time_constant, offset]),
            param_uncerts=np.array([-1, -1, -1])
        )

# **Sinusoidal Fit**
#
# $$f(x) = A \sin (2 \pi f x + \phi)$$

class SineModel(Model):
    def __init__(self, amplitude, frequency, phase):
        def fit_function(x, amplitude, freq, phase):
            return amplitude * np.sin(2.0 * np.pi * freq * x + phase)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Frequency', 'Phase'],
            param_values=np.array([amplitude, frequency, phase]),
            param_uncerts=np.array([-1, -1, -1])
        )

# **Offset Sinusoidal Fit**
#
# $$f(x) = A \sin (2 \pi f x + \phi) + C$$

class OffsetSineModel(Model):
    def __init__(self, amplitude, frequency, phase, offset):
        def fit_function(x, amplitude, freq, phase, offset):
            return amplitude * np.sin(2.0 * np.pi * freq * x + phase) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Frequency', 'Phase', 'Offset'],
            param_values=np.array([amplitude, frequency, phase, offset]),
            param_uncerts=np.array([-1, -1, -1, -1])
        )

# **Ringdown Fit**
#
# $$f(x) = A \exp(-x/\tau) \cos(2 \pi f_0 x + \phi)$$

class RingdownModel(Model):
    def __init__(self, amplitude, time_constant, resonant_frequency, phase):
        def fit_function(x, amplitude, time_constant, resonant_frequency, phase):
            return amplitude * np.exp(-x/time_constant) * \
                np.cos(2.0 * np.pi * resonant_frequency * x + phase)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Resonant Frequency', 'Phase'],
            param_values=np.array([amplitude, time_constant, resonant_frequency, phase]),
            param_uncerts=np.array([-1, -1, -1, -1])
        )

# **RC Response Model**
# $$V(f) = V_0 / \sqrt{1 + (2\pi f \tau)^2} + C$$

class RCResponseModel(Model):
    def __init__(self, amplitude, time_constant, offset):
        def fit_function(x, amplitude, time_constant, offset):
            return amplitude / np.sqrt(1 + (2 * np.pi * x * time_constant) ** 2) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Offset'],
            param_values=np.array([amplitude, time_constant, offset]),
            param_uncerts=np.array([-1, -1, -1])
        )

# **LRC Response Model**
# $$V(f) = V_0/\sqrt{1 + \left(\frac{2\pi}{\gamma f}\right)^2 (f^2 - f^2_0)^2}$$

class LRCResponseModel(Model):
    def __init__(self, amplitude, gamma, resonant_frequency):
        def fit_function(x, amplitude, gamma, resonant_frequency):
            return amplitude / np.sqrt(1 + (2 * np.pi / (gamma * x)) ** 2 *(x**2-resonant_frequency ** 2) ** 2)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Resonant Frequency'],
            param_values=np.array([amplitude, time_constant, resonant_frequency]),
            param_uncerts=np.array([-1, -1, -1])
        )

# **2-Parameter Linear Fit**
#
# $$f(x, m, b) = mx + b$$

class OffsetLinearModel(Model):
    def __init__(self, slope, intercept):
        def fit_function(x, slope, intercept):
            return slope * x + intercept
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Slope', 'Intercept'],
            param_values=np.array([slope, intercept]),
            param_uncerts=np.array([-1, -1])
        )


# **1-Parameter Linear Fit**
# $$f(x, m) = mx$$

class LinearModel(Model):
    def __init__(self, slope):
        def fit_function(x, slope):
            return slope * x
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Slope'],
            param_values=np.array([slope]),
            param_uncerts=np.array([-1])
        )


# **Custom Fit Model**

class CustomFitModel(Model):
    initial_params = None
    
    def __init__(self, fit_function: Callable, initial_params: dict):
        super().__init__(
            fit_function=fit_function,
            param_names=list(initial_params.keys()),
            param_values=np.array(list(initial_params.values())),
            param_uncerts=np.full(len(initial_params), -1.0)
        )


# +
@dataclass
class FitModelResult:
    initial_guess_graph:           Image = None
    initial_guess_residuals_graph: Image = None
    autofit_graph:                 Image = None
    autofit_residuals_graph:       Image = None

    chi2:              np.float64 = None
    covariance_matrix: np.array   = None
    
def print_results(model, results, print_cov=False, units=None):
    print(model.tabulate(units=units))
    print('Chi^2 = %.3f' % results.chi2)
        
    if print_cov:
        print("Covariance Values:")
        for i, fit_covariance in enumerate(results.covariance_matrix):
            for j in range(i+1, len(fit_covariance)):
                print(f"{model.param_names[i]} and {model.param_names[j]}: {results.covariance_matrix[i,j]:.3e}")
            print("\n")


# -

VOLTAGE_VERSUS_TIME_GRAPH_OPTIONS = GraphingOptions(
    x_label='Time',
    y_label='Voltage',
    x_units='s',
    y_units='V'
)

def plot_channels(ch1, ch2, graphing_options):
    plt.figure()
    plt.title('Channels 1 and 2')
    graphing_options.plot_data(ch1.x, ch1.y, ch1.dx, ch1.dy, label='Channel 1', color='lightblue')
    graphing_options.plot_data(ch2.x, ch2.y, ch2.dx, ch2.dy, label='Channel 2', color='orange')
    graphing_options.set_labels()
    plt.legend()
    plt.show()

def plot_channel_lissajous(ch1, ch2, graphing_options):
    plt.figure()
    plt.title('Channels 1 and 2 Lissajou')
    plt.errorbar(ch1.y, ch2.y, xerr=ch1.dy, yerr=ch2.dy, marker='.', linestyle='-')
    graphing_options.set_labels()
    plt.show()    

def plot_dataset(dataset, graphing_options):
    plt.figure()
    plt.title(graphing_options.default_title())
    graphing_options.plot_data(dataset.x, dataset.y, dataset.dx, dataset.dy)
    graphing_options.set_labels()
    plt.show()


def dataset_apply_arg(dataset, arg):
    dataset.x = dataset.x[arg]
    dataset.y = dataset.y[arg]
    dataset.dx = dataset.dx[arg]
    dataset.dy = dataset.dy[arg]

def sort_dataset(dataset):
    order = dataset.x.argsort()
    dataset_apply_arg(dataset, order)

def shear_dataset(dataset, n):
    if n == 0:
        return dataset
    dataset_apply_arg(dataset, slice(None, -n))

def trim_dataset(dataset, trim_range=None, graphing_options=None, plot=False):
    indices = np.arange(len(dataset.x))
    
    if trim_range is None:
        trim_range = (0, len(dataset.x))
    
    trimmed = Dataset(
        x  =  dataset.x[trim_range[0]:trim_range[1]],
        dx = dataset.dx[trim_range[0]:trim_range[1]],
        y  =  dataset.y[trim_range[0]:trim_range[1]],
        dy = dataset.dy[trim_range[0]:trim_range[1]]
    )
    
    if plot:
        plt.figure()
        plt.scatter(indices, dataset.y, marker='.')
        graphing_options.set_labels(xlabel='Index')
        mask = (indices >= trim_range[0]) & (indices <= trim_range[1])

        plt.fill_between(
            indices, min(dataset.y), max(dataset.y), 
            where = mask, 
            color='green', 
            alpha=0.1, 
            label='Trimmed Range'
        )

        plt.legend()
        plt.show()
        
    return trimmed


def load_channel(filename):
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    
    dx = np.zeros_like(xvalues)
    dy = np.zeros_like(yvalues)
    
    return Dataset(x=xvalues, y=yvalues, dx=dx, dy=dy)


def load_raw_data(filename, trim_range=None, plot=False, graphing_options=None):
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    indices = np.arange(len(xvalues))

    xvalues_trimmed = None
    yvalues_trimmed = None
    indices_trimmed = None
    
    if trim_range is not None:
        xvalues_trimmed = xvalues[trim_range[0]:trim_range[1]]
        yvalues_trimmed = yvalues[trim_range[0]:trim_range[1]]
        indices_trimmed = indices[trim_range[0]:trim_range[1]]

    if plot:
        plt.figure()
        plt.scatter(xvalues, yvalues, marker='.')
        graphing_options.set_labels()
        plt.title('Raw Data: ' + graphing_options.default_title())
        plt.show()

        plt.figure()
        plt.scatter(indices, yvalues, marker='.')
        graphing_options.set_labels(xlabel='Index')
        plt.title(f'Raw Data: {graphing_options.y_label} vs. Index, Round {graphing_options.data_round}')
        if trim_range is not None:
            mask = (indices > trim_range[0]) & (indices < trim_range[1])
            plt.fill_between(indices, min(yvalues), max(yvalues), 
                             where = mask, 
                             color='green', alpha=0.1, label='Trimmed Range')
            plt.legend()
        plt.show()

    return (xvalues,yvalues),(xvalues_trimmed,yvalues_trimmed)

def calculate_uncertainty(raw_data, method="default",
                          manual_uncert=None,
                          indices_range=None, y_range=None, 
                          plot=False, graphing_options=None):
    if indices_range is None:
        indices_range = (0, len(raw_data[0]))
    
    x_trimmed, y_trimmed = \
        map(lambda a: a[indices_range[0]:indices_range[1]],
            (raw_data[0], raw_data[1]))
    indices_trimmed = np.arange(0, len(x_trimmed))
    
    data_trimmed = (x_trimmed, y_trimmed, indices_trimmed)
    
    x, y = raw_data
    indices = np.arange(0, len(x))
    
    if plot:  
        plt.figure()
        plt.xlim(indices_range[0], indices_range[1])
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        graphing_options.set_labels(xlabel='Index')
        plt.scatter(indices, y, marker='.')
        plt.show()
        
        hist, bins = np.histogram(y_trimmed, bins=20)
        plt.bar(bins[:-1], hist, width = bins[1]-bins[0])
        plt.ylim(0, 1.2 * np.max(hist))
        plt.xlabel(f'Raw {graphing_options.y_label} Value ({graphing_options.y_units})')
        plt.ylabel('Number of Occurences')
        plt.show()
    
    match method:
        case "digital":
            digital = (np.max(y_trimmed) - np.min(y_trimmed)) / (2 * np.sqrt(3))
            print('Digital Uncertainty:', digital)
            return digital
        case "default":
            return isolate_noise_uncertainty(data_trimmed)
        case "manual":
            print('Manual Uncertainty:', manual_uncert)
            return manual_uncert


def isolate_noise_uncertainty(raw_data):
    x, y, indices = raw_data
    
    y_ave = np.mean(y)
    y_std = np.std(y)
    
    print('Mean = ', y_ave,y_std)
    print('Standard Deviation (Noise Value) = ', y_std)
    
    return y_std

def pack_data(data, uncertainty, p=100, trim_range=None, save=False, output_filename=None, plot=False, graphing_options=None):
    def pack(A, p):
        # A is an array, and p is the packing factor
        B = np.zeros(len(A)//p)
        i = 1
        while i - 1 < len(B):
            B[i-1] = np.mean(A[p*(i-1):p*i])
            i += 1
        return B
    
    x_raw, y_raw = data
    x = pack(x_raw, p)
    y = pack(y_raw, p)
    
    length = len(x)
    indices  = np.arange(length)
    x_uncert = np.zeros(length)
    
    y_uncert_raw  = uncertainty
    y_uncert_mean = y_uncert_raw / np.sqrt(p)
    y_uncert      = np.array([y_uncert_mean] * length)

    if trim_range is not None:
        x, x_uncert, y, y_uncert = \
            map(lambda a: a[trim_range[0]:trim_range[1]],
                (x, x_uncert, y, y_uncert))
        indices = np.arange(0, len(x))
        
    if plot:        
        plt.figure()
        graphing_options.plot_data(indices, y, x_uncert, y_uncert)
        graphing_options.set_labels()
        plt.xlabel('Index')
        plt.title('Packed Data')
        plt.show()

    if save:
        header = [np.array(['Time',  'u[time]', 'Voltage', 'u[Voltage]']), 
                  np.array(['(sec)', '(sec)',   '(V)',     '(V)'])]
        df = pd.DataFrame(np.array([x, x_uncert, y, y_uncert]).transpose(), columns=header)   
        
        csv_data = df.to_csv(output_filename, index = False)
        print('Packed Data Stored in ', output_filename)
    
    return Dataset(x=x, dx=x_uncert, y=y, dy=y_uncert)

# +
def calculate_chi_squared(fit_function, fit_params, x, y, sigma):
    dof = len(x) - len(fit_params)
    return np.sum((y - fit_function(x, *fit_params)) ** 2 / sigma ** 2) / dof

def calculate_t_score(a, da, b, db):
    return np.abs(a - b) / np.sqrt(da ** 2 + db ** 2)


# -

def autofit(data: Dataset, model: Model, graphing_options: GraphingOptions):
    results      = FitModelResult()
    fit_function = model.fit_function
    guesses      = model.values()
    
    # Theoretical x and y values for the sake of plotting
    guess_model_x = np.linspace(min(data.x), max(data.x), 500)
    guess_model_y  = fit_function(guess_model_x, *guesses)
    
    plt.figure()
    graphing_options.set_labels()
    plt.title('Initial Parameter Guess')
    graphing_options.plot_data(data.x, data.y, data.dx, data.dy, label='Measured Data')
    graphing_options.plot_model(guess_model_x, guess_model_y)
    plt.legend(loc="best", numpoints=1)
    results.initial_guess_graph = graphing_options.save_graph_and_close()

    # Residuals
    guess_y_fit = fit_function(data.x, *guesses)
    guess_residuals = data.y - guess_y_fit

    # Plot the residuals
    plt.figure()
    graphing_options.set_labels()
    plt.title("Residuals of Initial Parameter Guess")
    graphing_options.plot_residuals(data.x, guess_residuals, data.dy)
    results.initial_guess_residuals_graph = graphing_options.save_graph_and_close()

    # Perform the fit
    kwargs = dict(
        p0 = guesses,
        absolute_sigma = True,
        maxfev = int(1e5),
        sigma = data.dy
    )
    
    if model.has_bounds():
        kwargs["bounds"] = model.bounds()
    
    fit_params, fit_cov = curve_fit(fit_function, data.x, data.y, **kwargs)
    fit_params_error = np.sqrt(np.diag(fit_cov))
    
    # Store the fit results
    model.update_fit_results(fit_params, fit_params_error)
    
    results.chi2 = calculate_chi_squared(fit_function, fit_params, data.x, data.y, data.dy)
    results.covariance_matrix = fit_cov

    # Evaluate the Autofit
    
    model_x = np.linspace(min(data.x), max(data.x), len(data.x))
    model_y = fit_function(model_x, *fit_params)
    y_fit = fit_function(data.x, *fit_params)
    residuals = data.y - y_fit

    plt.figure()
    graphing_options.set_labels()
    plt.title('Best Fit of Function to Data')
    graphing_options.plot_data(data.x, data.y, data.dx, data.dy, label='Measured Data');
    graphing_options.plot_model(model_x, model_y);
    plt.legend(loc='best',numpoints=1)
    results.autofit_graph = graphing_options.save_graph_and_close()
    
    # The residuals plot
    plt.figure()
    graphing_options.set_labels()
    graphing_options.plot_residuals(data.x, residuals, data.dy)
    results.autofit_residuals_graph = graphing_options.save_graph_and_close()

    return results
