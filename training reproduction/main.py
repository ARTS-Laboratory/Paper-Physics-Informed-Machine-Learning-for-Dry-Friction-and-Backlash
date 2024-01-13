from train_experimental_model import train_experimental_model_loop
from train_numerical_model import train_numerical_model
from run_validation import run_numerical, run_experimental
from analyze_validation import analyze_experimental, analyze_numerical
from plots import experimental_harmonic_plot,\
                  experimental_earthquake_plot,\
                  experimental_earthquake_displacement_plot,\
                  experimental_state_space_plot,\
                  experimental_sigma_0_in_time_plot,\
                  numerical_harmonic_plot,\
                  numerical_earthquake_plot,\
                  numerical_state_space_plot,\
                  numerical_sigma_0_in_time_plot
"""
run all scripts
"""
if __name__ == '__main__':
    train_experimental_model_loop()
    train_numerical_model()
    
    run_experimental()
    run_numerical()
    analyze_experimental()
    analyze_numerical()
    #%% plots
    experimental_harmonic_plot()
    experimental_earthquake_plot()
    experimental_earthquake_displacement_plot()
    experimental_state_space_plot()
    experimental_sigma_0_in_time_plot()
    numerical_harmonic_plot()
    numerical_earthquake_plot()
    numerical_state_space_plot()
    numerical_sigma_0_in_time_plot()