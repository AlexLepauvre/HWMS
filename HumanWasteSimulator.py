import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
import pandas as pd
import imageio
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 1. ToiletConfiguration Class
class ToiletConfiguration:
    def __init__(self, num_normal_toilets, num_pee_buckets, 
                 volume_normal_toilet, volume_pee_bucket):
        """
        Initializes the toilet configuration.
        
        Parameters:
        - num_normal_toilets: Number of regular toilets.
        - num_pee_buckets: Number of pee buckets for men.
        - volume_normal_toilet: Volume capacity of each normal toilet (Liters).
        - volume_pee_bucket: Volume capacity of each pee bucket (Liters).
        """
        self.num_normal_toilets = num_normal_toilets
        self.num_pee_buckets = num_pee_buckets
        self.volume_normal_toilet = volume_normal_toilet
        self.volume_pee_bucket = volume_pee_bucket
        
        # Total capacities
        self.total_normal_toilet_capacity = num_normal_toilets * volume_normal_toilet
        self.total_pee_bucket_capacity = num_pee_buckets * volume_pee_bucket

# 2. Guest Class
class Guest:
    def __init__(self, gender, is_party_pooper, will_bush_peeing, 
                 pee_volume, poo_volume, pee_frequency, poo_frequency):
        """
        Initializes a guest.
        
        Parameters:
        - gender: 'male' or 'female'.
        - is_party_pooper: Boolean indicating if the guest will poo at the party.
        - will_bush_peeing: Boolean indicating if a male guest will bush pee if buckets are full.
        - pee_volume: Volume of pee per sitting (Liters).
        - poo_volume: Volume of poo per sitting (Liters).
        - pee_frequency: Number of pee events per day.
        - poo_frequency: Number of poo events per day.
        """
        self.gender = gender
        self.is_party_pooper = is_party_pooper
        self.will_bush_peeing = will_bush_peeing if gender == 'male' else False
        self.pee_volume = max(pee_volume, 0.0)  # Ensure non-negative
        self.poo_volume = max(poo_volume, 0.0)  # Ensure non-negative
        self.pee_frequency = max(pee_frequency, 0.0)  # Ensure non-negative
        self.poo_frequency = max(poo_frequency, 0.0)  # Ensure non-negative

    def should_produce_pee(self, time_step_hours):
        """
        Determines if the guest produces pee in the current time step.
        
        Parameters:
        - time_step_hours: Duration of the time step in hours.
        
        Returns:
        - Boolean indicating if pee is produced.
        """
        if self.pee_frequency == 0:
            return False
        # Convert frequency per day to probability per hour
        lambda_pee = self.pee_frequency / 24
        # Probability of producing pee in this time step
        probability = 1 - np.exp(-lambda_pee * time_step_hours)
        return np.random.rand() < probability

    def should_produce_poo(self, time_step_hours):
        """
        Determines if the guest produces poo in the current time step.
        
        Parameters:
        - time_step_hours: Duration of the time step in hours.
        
        Returns:
        - Boolean indicating if poo is produced.
        """
        if not self.is_party_pooper or self.poo_frequency == 0:
            return False
        # Convert frequency per day to probability per hour
        lambda_poo = self.poo_frequency / 24
        # Probability of producing poo in this time step
        probability = 1 - np.exp(-lambda_poo * time_step_hours)
        return np.random.rand() < probability

    def produce_pee(self, time_step_hours):
        """
        Simulates pee production for this guest over one time step.
        
        Returns:
        - Pee volume (Liters) if produced, else 0.0.
        """
        if self.should_produce_pee(time_step_hours):
            return self.pee_volume
        return 0.0

    def produce_poo(self, time_step_hours):
        """
        Simulates poo production for this guest over one time step.
        
        Returns:
        - Poo volume (Liters) if produced, else 0.0.
        """
        if self.should_produce_poo(time_step_hours):
            return self.poo_volume
        return 0.0

# 3. Crowd Class
class Crowd:
    def __init__(self, total_people, ratio_male_female, 
                 ratio_party_pooper, ratio_bush_peeing, 
                 mean_pee_volume, std_pee_volume, 
                 mean_poo_volume, std_poo_volume,
                 mean_pee_frequency, std_pee_frequency,
                 mean_poo_frequency, std_poo_frequency):
        """
        Initializes the crowd.
        
        Parameters:
        - total_people: Total number of guests.
        - ratio_male_female: Proportion of males (e.g., 0.5 for 50% males).
        - ratio_party_pooper: Proportion of guests who will poo at the party.
        - ratio_bush_peeing: Proportion of males who will bush pee if buckets are full.
        - mean_pee_volume: Mean volume of pee per event (Liters).
        - std_pee_volume: Standard deviation of pee volume.
        - mean_poo_volume: Mean volume of poo per event (Liters).
        - std_poo_volume: Standard deviation of poo volume.
        - mean_pee_frequency: Mean number of pee events per day.
        - std_pee_frequency: Standard deviation of pee frequency.
        - mean_poo_frequency: Mean number of poo events per day.
        - std_poo_frequency: Standard deviation of poo frequency.
        """
        self.total_people = total_people
        self.ratio_male_female = ratio_male_female
        self.ratio_party_pooper = ratio_party_pooper
        self.ratio_bush_peeing = ratio_bush_peeing
        self.mean_pee_volume = mean_pee_volume
        self.std_pee_volume = std_pee_volume
        self.mean_poo_volume = mean_poo_volume
        self.std_poo_volume = std_poo_volume
        self.mean_pee_frequency = mean_pee_frequency
        self.std_pee_frequency = std_pee_frequency
        self.mean_poo_frequency = mean_poo_frequency
        self.std_poo_frequency = std_poo_frequency
        
        self.guests = self.generate_guests()
        
    def generate_guests(self):
        """
        Generates the list of Guest objects based on the crowd configuration.
        
        Returns:
        - List of Guest objects.
        """
        guests = []
        num_men = int(self.total_people * self.ratio_male_female)
        num_women = self.total_people - num_men
        
        for _ in range(num_men):
            is_party_pooper = np.random.rand() < self.ratio_party_pooper
            will_bush_peeing = np.random.rand() < self.ratio_bush_peeing
            pee_volume = max(np.random.normal(self.mean_pee_volume, self.std_pee_volume), 0.01)  # Avoid zero or negative
            poo_volume = max(np.random.normal(self.mean_poo_volume, self.std_poo_volume), 0.0) if is_party_pooper else 0.0
            pee_frequency = max(np.random.normal(self.mean_pee_frequency, self.std_pee_frequency), 0.0)
            poo_frequency = max(np.random.normal(self.mean_poo_frequency, self.std_poo_frequency), 0.0) if is_party_pooper else 0.0
            guests.append(Guest('male', is_party_pooper, will_bush_peeing, 
                                pee_volume, poo_volume, pee_frequency, poo_frequency))
        
        for _ in range(num_women):
            is_party_pooper = np.random.rand() < self.ratio_party_pooper
            # Women don't bush pee
            pee_volume = max(np.random.normal(self.mean_pee_volume, self.std_pee_volume), 0.01)
            poo_volume = max(np.random.normal(self.mean_poo_volume, self.std_poo_volume), 0.0) if is_party_pooper else 0.0
            pee_frequency = max(np.random.normal(self.mean_pee_frequency, self.std_pee_frequency), 0.0)
            poo_frequency = max(np.random.normal(self.mean_poo_frequency, self.std_poo_frequency), 0.0) if is_party_pooper else 0.0
            guests.append(Guest('female', is_party_pooper, False, 
                                pee_volume, poo_volume, pee_frequency, poo_frequency))
        
        return guests

# 4. Simulation Class with Progressive Plotting and Confidence Intervals
class Simulation:
    def __init__(self, toilet_config, crowd_params, sawdust_ratio=0.5, 
                 time_step=1, total_time=24, num_bootstrap=1000, 
                 gif_filename='simulation_with_ci.gif',
                 start_hour=18):
        """
        Initializes the simulation.
        
        Parameters:
        - toilet_config: Instance of ToiletConfiguration.
        - crowd_params: Dictionary of parameters for Crowd class.
        - sawdust_ratio: Proportion of sawdust added to waste volume.
        - time_step: Time step for the simulation (hours).
        - total_time: Total time to simulate (hours).
        - num_bootstrap: Number of bootstrap simulations for CIs.
        - gif_filename: Filename for the saved GIF.
        - start_hour: Starting hour of the simulation (0-23). Default is 18 (6 PM).
        """
        self.toilet_config = toilet_config
        self.crowd_params = crowd_params
        self.sawdust_ratio = sawdust_ratio
        self.time_step = time_step
        self.total_time = total_time
        self.num_bootstrap = num_bootstrap
        self.gif_filename = gif_filename
        self.start_hour = start_hour  # New parameter for start time
        
        # Time points
        self.time_points = np.arange(0, self.total_time + self.time_step, self.time_step)
        
        # Initialize DataFrames to store simulation results
        self.df_pee_buckets = pd.DataFrame(columns=self.time_points)
        self.df_normal_toilets = pd.DataFrame(columns=self.time_points)
        self.df_total_volumes = pd.DataFrame(columns=self.time_points)
        
        # For GIF frames
        self.frames = []
        
    def run_single_simulation(self, seed=None):
        """
        Runs a single simulation and returns the time series data.
        
        Parameters:
        - seed: Optional random seed for reproducibility.
        
        Returns:
        - Dictionary containing time points and waste volumes.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize Crowd
        crowd = Crowd(
            total_people=self.crowd_params['total_people'],
            ratio_male_female=self.crowd_params['ratio_male_female'],
            ratio_party_pooper=self.crowd_params['ratio_party_pooper'],
            ratio_bush_peeing=self.crowd_params['ratio_bush_peeing'],
            mean_pee_volume=self.crowd_params['mean_pee_volume'],
            std_pee_volume=self.crowd_params['std_pee_volume'],
            mean_poo_volume=self.crowd_params['mean_poo_volume'],
            std_poo_volume=self.crowd_params['std_poo_volume'],
            mean_pee_frequency=self.crowd_params['mean_pee_frequency'],
            std_pee_frequency=self.crowd_params['std_pee_frequency'],
            mean_poo_frequency=self.crowd_params['mean_poo_frequency'],
            std_poo_frequency=self.crowd_params['std_poo_frequency']
        )
        
        # Initialize Simulation Volumes
        current_pee_bucket_volume = 0.0
        current_normal_toilet_volume = 0.0
        
        # Tracking Volumes
        pee_buckets = [0.0]
        normal_toilets = [0.0]
        total_volumes = [0.0]
        
        # Simulation Loop
        for t in self.time_points[1:]:
            # Waste production for this time step
            pee_to_toilets = 0.0
            poo_to_toilets = 0.0
            
            for guest in crowd.guests:
                # Pee production
                pee = guest.produce_pee(self.time_step)
                if pee > 0:
                    if guest.gender == 'male':
                        # Try to add to pee buckets
                        pee_with_sawdust = pee * (1 + self.sawdust_ratio)
                        if current_pee_bucket_volume + pee_with_sawdust <= self.toilet_config.total_pee_bucket_capacity:
                            current_pee_bucket_volume += pee_with_sawdust
                        else:
                            # Buckets are full or will be full; handle overflow
                            if guest.will_bush_peeing:
                                # Guest goes to bush; pee not added to any container
                                continue
                            else:
                                # Add to normal toilets
                                pee_to_toilets += pee_with_sawdust
                    else:
                        # Female guests always use normal toilets
                        pee_with_sawdust = pee * (1 + self.sawdust_ratio)
                        pee_to_toilets += pee_with_sawdust
                
                # Poo production
                poo = guest.produce_poo(self.time_step)
                if poo > 0:
                    poo_with_sawdust = poo * (1 + self.sawdust_ratio)
                    poo_to_toilets += poo_with_sawdust
            
            # Add pee and poo to normal toilets
            current_normal_toilet_volume += pee_to_toilets + poo_to_toilets
            
            # Handle overflow in normal toilets
            if current_normal_toilet_volume > self.toilet_config.total_normal_toilet_capacity:
                # Overflow occurs; cap at capacity
                current_normal_toilet_volume = self.toilet_config.total_normal_toilet_capacity
            
            # Record Volumes
            pee_buckets.append(current_pee_bucket_volume)
            normal_toilets.append(current_normal_toilet_volume)
            total_volumes.append(current_pee_bucket_volume + current_normal_toilet_volume)
        
        return {
            'pee_bucket_volumes': pee_buckets,
            'normal_toilet_volumes': normal_toilets,
            'total_volumes': total_volumes
        }
    
    def run_bootstrap_simulations(self):
        """
        Runs multiple simulations in parallel and stores the results.
        """
        print(f"Running {self.num_bootstrap} bootstrap simulations...")
        
        # Use multiprocessing Pool to run simulations in parallel
        with Pool() as pool:
            # Optionally, set different seeds for reproducibility
            seeds = np.random.randint(0, 1e6, size=self.num_bootstrap)
            results = pool.starmap(self.run_single_simulation, [(seed,) for seed in seeds])
        
        # Populate DataFrames
        for res in results:
            self.df_pee_buckets = self.df_pee_buckets._append(pd.Series(res['pee_bucket_volumes'], index=self.time_points), ignore_index=True)
            self.df_normal_toilets = self.df_normal_toilets._append(pd.Series(res['normal_toilet_volumes'], index=self.time_points), ignore_index=True)
            self.df_total_volumes = self.df_total_volumes._append(pd.Series(res['total_volumes'], index=self.time_points), ignore_index=True)
        
        print("Bootstrap simulations completed.")
    
    def compute_statistics(self):
        """
        Computes mean and 95% confidence intervals for each waste volume category at each time point.
        """
        print("Computing statistics...")
        
        # Compute means
        self.mean_pee_buckets = self.df_pee_buckets.mean()
        self.mean_normal_toilets = self.df_normal_toilets.mean()
        self.mean_total_volumes = self.df_total_volumes.mean()
        
        # Compute 95% confidence intervals using the percentile method
        self.lower_ci_pee_buckets = self.df_pee_buckets.quantile(0.025)
        self.upper_ci_pee_buckets = self.df_pee_buckets.quantile(0.975)
        
        self.lower_ci_normal_toilets = self.df_normal_toilets.quantile(0.025)
        self.upper_ci_normal_toilets = self.df_normal_toilets.quantile(0.975)
        
        self.lower_ci_total_volumes = self.df_total_volumes.quantile(0.025)
        self.upper_ci_total_volumes = self.df_total_volumes.quantile(0.975)
        
        print("Statistics computed.")
    
    def animate_progression(self):
        """
        Creates an animated plot showing the progression of mean waste accumulation with 95% CIs.
        Saves the animation as a GIF.
        """
        print("Creating animated plot with confidence intervals...")
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Volume (Liters)')
        ax.set_title('Waste Accumulation Over Time with 95% Confidence Intervals')
        ax.grid(True)
        
        # Plot capacity lines
        ax.axhline(y=self.toilet_config.total_pee_bucket_capacity, color='blue', linestyle=':', label="Pee Buckets Capacity")
        ax.axhline(y=self.toilet_config.total_normal_toilet_capacity, color='orange', linestyle=':', label="Normal Toilets Capacity")
        
        # Initialize lines for mean
        line_pee, = ax.plot([], [], color='blue', label="Mean Pee Buckets")
        line_normal, = ax.plot([], [], color='orange', label="Mean Normal Toilets")
        line_total, = ax.plot([], [], color='purple', linestyle='--', label="Mean Total Volume")
        
        # Initialize fill_between for CIs using mutable containers
        fill_pee = [None]
        fill_normal = [None]
        fill_total = [None]
        
        ax.legend()
        
        # Prepare data
        time_points = self.time_points
        num_frames = len(time_points)
        
        # Generate time labels starting at start_hour
        def generate_time_labels(start_hour, num_steps, step_size):
            """
            Generates a list of time labels starting at start_hour.
            
            Parameters:
            - start_hour: Starting hour (0-23).
            - num_steps: Number of steps.
            - step_size: Size of each step in hours.
            
            Returns:
            - List of formatted time strings (e.g., '18:00').
            """
            time_labels = []
            current_time = datetime(2000, 1, 1, start_hour, 0)  # Arbitrary date
            for _ in range(num_steps):
                time_labels.append(current_time.strftime('%H:%M'))
                current_time += timedelta(hours=step_size)
                if current_time.hour >= 24:
                    current_time = current_time.replace(hour=current_time.hour % 24)
            return time_labels
        
        time_labels = generate_time_labels(self.start_hour, len(time_points), self.time_step)
        
        # Function to update each frame
        def update(frame):
            current_time = time_points[:frame+1]
            current_labels = time_labels[:frame+1]
            
            # Update mean lines using iloc for position-based slicing
            line_pee.set_data(current_time, self.mean_pee_buckets.iloc[:frame+1])
            line_normal.set_data(current_time, self.mean_normal_toilets.iloc[:frame+1])
            line_total.set_data(current_time, self.mean_total_volumes.iloc[:frame+1])
            
            # Remove previous fill_between patches if they exist
            if fill_pee[0]:
                fill_pee[0].remove()
            if fill_normal[0]:
                fill_normal[0].remove()
            if fill_total[0]:
                fill_total[0].remove()
            
            # Update confidence intervals using iloc for position-based slicing
            fill_pee[0] = ax.fill_between(
                current_time, 
                self.lower_ci_pee_buckets.iloc[:frame+1], 
                self.upper_ci_pee_buckets.iloc[:frame+1], 
                color='blue', alpha=0.2
            )
            fill_normal[0] = ax.fill_between(
                current_time, 
                self.lower_ci_normal_toilets.iloc[:frame+1], 
                self.upper_ci_normal_toilets.iloc[:frame+1], 
                color='orange', alpha=0.2
            )
            fill_total[0] = ax.fill_between(
                current_time, 
                self.lower_ci_total_volumes.iloc[:frame+1], 
                self.upper_ci_total_volumes.iloc[:frame+1], 
                color='purple', alpha=0.2
            )
            
            # Adjust axes limits if necessary
            ax.set_xlim(0, self.total_time)
            max_volume = max(
                self.upper_ci_pee_buckets.max(),
                self.upper_ci_normal_toilets.max(),
                self.upper_ci_total_volumes.max(),
                self.toilet_config.total_normal_toilet_capacity,
                self.toilet_config.total_pee_bucket_capacity
            )
            ax.set_ylim(0, max_volume * 1.1)
            
            # Update x-axis labels
            if frame < num_frames:
                ax.set_xticks(current_time)
                ax.set_xticklabels(current_labels, rotation=45)
            
            return [line_pee, line_normal, line_total, fill_pee[0], fill_normal[0], fill_total[0]]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames, interval=500, blit=False)
        
        # Save as GIF
        anim.save(self.gif_filename, writer='pillow', fps=2)
        plt.close(fig)
        
        print(f"Animated GIF saved as '{self.gif_filename}'.")
    
    def run(self):
        """
        Executes the complete simulation process:
        1. Runs bootstrap simulations.
        2. Computes statistics.
        3. Creates and saves the animated GIF.
        """
        self.run_bootstrap_simulations()
        self.compute_statistics()
        self.animate_progression()
        print("Simulation run completed.")

# Example Usage
if __name__ == "__main__":
    # Define Toilet Configuration
    toilet_config = ToiletConfiguration(
        num_normal_toilets=4,
        num_pee_buckets=3,
        volume_normal_toilet=60,  # Liters per normal toilet
        volume_pee_bucket=60      # Liters per pee bucket
    )
    
    # Define Crowd Parameters
    crowd_params = {
        'total_people': 250,
        'ratio_male_female': 0.5,             # 50% males
        'ratio_party_pooper': 0.5,            # 50% will poo
        'ratio_bush_peeing': 0.5,             # 50% of males will bush pee if buckets are full
        'mean_pee_volume': 0.3,               # Mean pee volume per event (Liters)
        'std_pee_volume': 0.1,               # Std dev of pee volume
        'mean_poo_volume': 0.2,               # Mean poo volume per event (Liters)
        'std_poo_volume': 0.03,               # Std dev of poo volume
        'mean_pee_frequency': 7,              # Mean pee events per day
        'std_pee_frequency': 2,               # Std dev of pee frequency
        'mean_poo_frequency': 1,              # Mean poo events per day
        'std_poo_frequency': 0.5               # Std dev of poo frequency
    }
    
    # Initialize Simulation
    simulation = Simulation(
        toilet_config=toilet_config,
        crowd_params=crowd_params,
        sawdust_ratio=0.5,                   # 50% sawdust
        time_step=0.4,                          # 1 hour per step
        total_time=24,                        # Simulate for 24 hours
        num_bootstrap=1000,                   # Number of simulations for CIs
        gif_filename='simulation_with_ci.gif',  # Output GIF filename
        start_hour=18                          # Start at 18:00
    )
    
    # Run the Simulation
    simulation.run()
    
    # Print Final Aggregated Results
    print("\nFinal Aggregated Results:")
    print(f"Mean Volume in Men's Pee Buckets: {simulation.mean_pee_buckets.iloc[-1]:.2f} Liters ± {(simulation.upper_ci_pee_buckets.iloc[-1] - simulation.mean_pee_buckets.iloc[-1]):.2f} Liters (95% CI)")
    print(f"Mean Volume in Normal Toilets: {simulation.mean_normal_toilets.iloc[-1]:.2f} Liters ± {(simulation.upper_ci_normal_toilets.iloc[-1] - simulation.mean_normal_toilets.iloc[-1]):.2f} Liters (95% CI)")
    print(f"Mean Total Volume of Waste: {simulation.mean_total_volumes.iloc[-1]:.2f} Liters ± {(simulation.upper_ci_total_volumes.iloc[-1] - simulation.mean_total_volumes.iloc[-1]):.2f} Liters (95% CI)")
