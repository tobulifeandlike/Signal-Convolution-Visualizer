import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox
from IPython.display import HTML

class ConvolutionVisualizer:
    """
    Interactive convolution visualizer for discrete signals
    Shows x[k], h[n-k], and their convolution y[n]
    """

    def __init__(self):
        # Initialize default signals
        self.x_signal = np.array([1, 2, 3, 2, 1])  # Default x[k]
        self.h_signal = np.array([1, 1, 1])  # Default h[k]
        self.n_current = 0  # Current time index
        self.n_max = 20  # Maximum time index for animation
        self.N_display = 10  # Display range for plotting
        self.is_playing = False  # Animation state

        # Compute initial convolution
        self.compute_convolution()

        # Create figure and subplots
        self.setup_figure()

        # Create interactive widgets
        self.setup_widgets()

        # Initial plot
        self.update_plot(0)

    def compute_convolution(self):
        """
        Compute full convolution y[n] = x[k] * h[k]
        Length of convolution = len(x) + len(h) - 1
        """
        # Use numpy convolve function for convolution
        self.y_conv = np.convolve(self.x_signal, self.h_signal)

    def compute_h_flipped(self, n):
        """
        Compute h[n-k] for given time index n
        This is h[k] time-reversed and shifted by n
        """
        # Create array for h[n-k] over display range
        k_range = np.arange(-self.N_display, self.N_display + 1)
        h_flipped = np.zeros_like(k_range, dtype=float)

        # For each k, compute index for h[n-k]
        for i, k in enumerate(k_range):
            h_index = n - k  # h[n-k] means index is (n-k)
            # Check if h_index is within valid range
            if 0 <= h_index < len(self.h_signal):
                h_flipped[i] = self.h_signal[h_index]
            # else: remains 0 (zero padding)

        return k_range, h_flipped

    def compute_convolution_at_n(self, n):
        """
        Compute convolution value at specific time n
        y[n] = sum over k: x[k] * h[n-k]
        """
        conv_sum = 0.0
        # Sum over all k where x[k] exists
        for k in range(len(self.x_signal)):
            h_index = n - k  # Index for h[n-k]
            # Check if h_index is valid
            if 0 <= h_index < len(self.h_signal):
                conv_sum += self.x_signal[k] * self.h_signal[h_index]

        return conv_sum

    def setup_figure(self):
        """
        Create figure with 3 subplots and space for widgets
        """
        # Create figure with specific size
        self.fig = plt.figure(figsize=(14, 12))

        # Create grid for subplots (leave space at bottom for widgets)
        # 3 rows for plots, additional space for widgets
        self.ax1 = plt.subplot(3, 1, 1)  # x[k] subplot
        self.ax2 = plt.subplot(3, 1, 2)  # h[n-k] subplot
        self.ax3 = plt.subplot(3, 1, 3)  # y[n] subplot

        # Set titles for each subplot
        self.ax1.set_title('Signal x[k]', fontsize=6, fontweight='bold')
        self.ax2.set_title('Signal h[n-k] (time-reversed and shifted)', fontsize=6, fontweight='bold')
        self.ax3.set_title('Convolution y[n] = x[k] * h[k]', fontsize=6, fontweight='bold')

        # Set labels
        self.ax1.set_xlabel('k',fontsize = 5)
        self.ax1.set_ylabel('x[k]')
        self.ax2.set_xlabel('k',fontsize = 5)
        self.ax2.set_ylabel('h[n-k]')
        self.ax3.set_xlabel('n',fontsize = 5)
        self.ax3.set_ylabel('y[n]')

        # Add grid to all subplots
        self.ax1.grid(True, alpha=0.3)
        self.ax2.grid(True, alpha=0.3)
        self.ax3.grid(True, alpha=0.3)

        # Adjust spacing between subplots
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.96, hspace=0.5)

    def setup_widgets(self):
        """
        Create interactive widgets: textboxes, sliders, buttons
        """
        # TextBox for x[k] signal input
        ax_text_x = plt.axes([0.15, 0.25, 0.3, 0.03])
        self.text_x = TextBox(ax_text_x, 'x[k]:', initial='1,2,3,2,1')
        self.text_x.on_submit(self.update_x_signal)

        # TextBox for h[k] signal input
        ax_text_h = plt.axes([0.15, 0.20, 0.3, 0.03])
        self.text_h = TextBox(ax_text_h, 'h[k]:', initial='1,1,1')
        self.text_h.on_submit(self.update_h_signal)

        # Slider for time index n
        ax_slider_n = plt.axes([0.15, 0.14, 0.7, 0.02])
        self.slider_n = Slider(ax_slider_n, 'Time n', 0, self.n_max,
                               valinit=0, valstep=1)
        self.slider_n.on_changed(self.update_n)

        # Slider for n_max
        ax_slider_nmax = plt.axes([0.15, 0.10, 0.7, 0.02])
        self.slider_nmax = Slider(ax_slider_nmax, 'n_max', 1, 50,
                                  valinit=self.n_max, valstep=1)
        self.slider_nmax.on_changed(self.update_nmax)

        # Slider for display range N
        ax_slider_N = plt.axes([0.15, 0.06, 0.7, 0.02])
        self.slider_N = Slider(ax_slider_N, 'Display N', 5, 20,
                               valinit=self.N_display, valstep=1)
        self.slider_N.on_changed(self.update_N)

        # Play/Pause button
        ax_button_play = plt.axes([0.55, 0.25, 0.1, 0.04])
        self.button_play = Button(ax_button_play, 'Play')
        self.button_play.on_clicked(self.toggle_play)

        # Reset button
        ax_button_reset = plt.axes([0.7, 0.25, 0.1, 0.04])
        self.button_reset = Button(ax_button_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_animation)

    def stem_plot(self, ax, k_values, signal_values, color='blue'):
        """
        Create stem plot for discrete signal
        Draws vertical lines with markers at the top
        """
        # Clear previous plot
        ax.clear()

        # Draw stems (vertical lines)
        ax.stem(k_values, signal_values, linefmt=color, markerfmt='o',
                basefmt='k-')

        # Draw horizontal axis at y=0
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Add grid
        ax.grid(True, alpha=0.3)

    def update_plot(self, frame):
        """
        Update all three subplots for current time index n
        Called by animation or slider
        """
        # Get current time index
        n = self.n_current

        # Plot 1: x[k] signal (fixed)
        k_x = np.arange(len(self.x_signal))
        self.stem_plot(self.ax1, k_x, self.x_signal, color='blue')
        self.ax1.set_title('Signal x[k]', fontsize=6, fontweight='bold')
        self.ax1.set_xlabel('k',fontsize = 5)
        self.ax1.set_ylabel('x[k]')
        self.ax1.set_xlim(-self.N_display, self.N_display)

        # Plot 2: h[n-k] signal (animated)
        k_range, h_flipped = self.compute_h_flipped(n)
        self.stem_plot(self.ax2, k_range, h_flipped, color='green')
        self.ax2.set_title(f'Signal h[n-k] at n={n}', fontsize=6, fontweight='bold')
        self.ax2.set_xlabel('k',fontsize = 5)
        self.ax2.set_ylabel('h[n-k]')
        self.ax2.set_xlim(-self.N_display, self.N_display)

        # Plot 3: Convolution result up to current n
        n_range = np.arange(min(n + 1, len(self.y_conv)))
        y_values = self.y_conv[:n + 1]
        self.stem_plot(self.ax3, n_range, y_values, color='red')
        self.ax3.set_title(f'Convolution y[n] up to n={n}', fontsize=6, fontweight='bold')
        self.ax3.set_xlabel('n',fontsize = 5)
        self.ax3.set_ylabel('y[n]')
        self.ax3.set_xlim(-self.N_display, self.N_display)

        # Add text showing current convolution value
        if n < len(self.y_conv):
            self.ax3.text(0.02, 0.95, f'y[{n}] = {self.y_conv[n]:.3f}',
                         transform=self.ax3.transAxes,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         verticalalignment='top')

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def update_x_signal(self, text):
        """
        Parse and update x[k] signal from text input
        """
        try:
            # Split by comma and convert to float array
            values = [float(v.strip()) for v in text.split(',')]
            self.x_signal = np.array(values)
            # Recompute convolution
            self.compute_convolution()
            # Reset time to 0
            self.n_current = 0
            self.slider_n.set_val(0)
            # Update plot
            self.update_plot(0)
        except:
            print("Invalid input for x[k]. Use comma-separated numbers.")

    def update_h_signal(self, text):
        """
        Parse and update h[k] signal from text input
        """
        try:
            # Split by comma and convert to float array
            values = [float(v.strip()) for v in text.split(',')]
            self.h_signal = np.array(values)
            # Recompute convolution
            self.compute_convolution()
            # Reset time to 0
            self.n_current = 0
            self.slider_n.set_val(0)
            # Update plot
            self.update_plot(0)
        except:
            print("Invalid input for h[k]. Use comma-separated numbers.")

    def update_n(self, val):
        """
        Update time index n from slider
        """
        # Get integer value from slider
        self.n_current = int(val)
        # Update plot
        self.update_plot(0)

    def update_nmax(self, val):
        """
        Update maximum time index n_max from slider
        """
        # Get integer value from slider
        self.n_max = int(val)
        # Update slider_n range
        self.slider_n.valmax = self.n_max
        self.slider_n.ax.set_xlim(0, self.n_max)

    def update_N(self, val):
        """
        Update display range N from slider
        """
        # Get integer value from slider
        self.N_display = int(val)
        # Update plot
        self.update_plot(0)

    def toggle_play(self, event):
        """
        Start or stop animation
        """
        if not self.is_playing:
            # Start animation
            self.is_playing = True
            self.button_play.label.set_text('Pause')
            # Create animation object
            self.anim = FuncAnimation(self.fig, self.animate_step,
                                     interval=500, blit=False)
        else:
            # Stop animation
            self.is_playing = False
            self.button_play.label.set_text('Play')
            # Stop animation object
            if hasattr(self, 'anim'):
                self.anim.event_source.stop()

    def animate_step(self, frame):
        """
        Single animation step - increment n and update plot
        """
        if self.is_playing:
            # Increment time index
            self.n_current += 1
            # Loop back to 0 if exceeding n_max
            if self.n_current > self.n_max:
                self.n_current = 0
            # Update slider position
            self.slider_n.set_val(self.n_current)
            # Update plot
            self.update_plot(frame)

    def reset_animation(self, event):
        """
        Reset time index to 0
        """
        # Stop animation if playing
        if self.is_playing:
            self.toggle_play(None)
        # Reset time to 0
        self.n_current = 0
        self.slider_n.set_val(0)
        # Update plot
        self.update_plot(0)

    def show(self):
        """
        Display the interactive plot
        """
        plt.show()

# Create and display the visualizer
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = ConvolutionVisualizer()
    # Display interactive plot
    visualizer.show()