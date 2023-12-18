import numpy as np
import matplotlib.pyplot as plt

# helper function for plotting
def plot(states):
    num_states = states.shape[0]
    time = np.arange(states.shape[1])

    plt.figure(figsize=(10, 2*num_states))

    names = ["position [m]","velocity [m/s]","acceleration [m/s²]","p_set [bar]","p_actual [bar]","x_target [m]","error [m]","reward","?"]
    colors = ["tab:blue","tab:orange","tab:purple","tab:green","tab:green","tab:grey","tab:red","tab:cyan","black"]

    for i in range(num_states):
        plt.subplot(num_states, 1, i + 1)
        plt.plot(time, states[i, :], label=names[i],color=colors[i])
        if i == 0:
            plt.plot(time, states[5, :], label=names[5],color=colors[5]) # x_target
        if i == 6: # error
            plt.fill_between(time, 0, states[i, :], color=colors[i], alpha=0.5)
        plt.ylabel(names[i])
        plt.legend()
        plt.grid(True)

    plt.xlabel('Time [steps]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_doc(states,filename):
    # plot 
    # position, with target and error
    # p_set, p_actual
    # reward
    num_states = states.shape[0]
    time = np.arange(states.shape[1])

    plt.figure(figsize=(4, 4))

    names = ["position [m]","velocity [m/s]","acceleration [m/s²]","p_set [bar]","p_actual [bar]","x_target [m]","error [m]","reward","?"]
    colors = ["tab:blue","tab:orange","tab:purple","tab:green","tab:green","tab:grey","tab:red","tab:cyan","black"]

    plt.subplot(3, 1, 1)
    plt.plot(time, states[5, :], label="x_target",color=colors[5]) # x_target
    plt.fill_between(time, states[5, :], states[0, :], color=colors[6], label="error [m]", alpha=0.3) # error
    plt.plot(time, states[0, :], label="position",color=colors[0]) # x
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel("x [m]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, states[3, :], label="p_set",color="tab:green") # p_set
    plt.plot(time, states[4, :], label="p_actual",color="darkgreen",linestyle='--') # p_actual
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel("p [bar]")
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, states[7, :], label="reward",color="tab:cyan") # reward
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.xlabel('Time [steps]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()