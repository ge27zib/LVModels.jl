using Plots, LinearAlgebra, SymbolicRegression

@doc """
    sparsify_data(sol; fraction=1.0)

Randomly sparsifies only the last dimension of the solution data from an `n`-dimensional system by masking a fraction of its values.

# Arguments
- `sol`: A solution object containing the time array (`sol.t`) and state values.
- `fraction`: The probability (between 0 and 1) of keeping a data point unmasked in the last dimension. Default is 1.0 (no sparsification).

# Returns
A tuple `(t, X_sparse, mask)`, where:
- `t`: The time array from the solution.
- `X_sparse`: The sparsified state data array, where only the last dimension is masked.
- `mask`: A binary mask array applied to the last dimension, indicating retained (`1`) and masked (`0`) values.
"""
function sparsify_data(sol; fraction=1.0)
    t = sol.t
    X = Array(sol)
    n_time = size(X, 2)

    # Generate a mask for the last dimension
    mask = rand(n_time) .< fraction  

    # Apply the mask only to the last dimension
    X_sparse = copy(X)
    X_sparse[end, :] .*= mask  # Mask only the last dimension

    return t, X_sparse, mask
end

@doc """
    get_mask_for_batch(batch_t, global_mask, t0, dt)

Extracts a slice of the global mask corresponding to a given batch time vector.

# Arguments
- `batch_t`: A vector of time points for the current batch.
- `global_mask`: A binary mask array corresponding to the full time series.
- `t0`: The starting time of the global time series.
- `dt`: The time step interval between consecutive points in the global time series.

# Returns
A mask slice corresponding to the time points in `batch_t`, extracted from `global_mask`.
"""
function get_mask_for_batch(batch_t, global_mask, t0, dt)
    indices = round.(Int, (batch_t .- t0) ./ dt .+ 1)
    return global_mask[indices]
end

@doc """
    loss(m, batch, truth, batch_mask; λ1=0f0, λ2=0f0)

Computes the loss for an n-dimensional system based on predicted values, ground truth, and a mask for the batch.
The loss consists of mean squared errors (MSE) for each dimension, along with regularization terms for model parameters.

# Arguments
- `m`: A model producing predictions for each dimension of the system.
- `batch`: The input data batch to the model.
- `truth`: The ground truth values for the n-dimensional system.
- `batch_mask`: A mask that is applied to the batch to handle missing or masked data points.
- `λ1`: The weight for the L1 regularization term (default is 0).
- `λ2`: The weight for the L2 regularization term (default is 0).

# Returns
The total loss, which is the sum of:
- Mean squared error (MSE) for each dimension (x, y, ..., n),
- L1 regularization of model parameters (if λ1 > 0),
- L2 regularization of model parameters (if λ2 > 0).
"""
function loss(m, batch, truth, batch_mask; λ1=0f0, λ2=0f0)
    pred = m(batch)
    n_dims = size(pred, 1)  
    total_loss = 0f0

    # Compute loss for each dimension
    for i in 1:n_dims
        if i == n_dims
            # Apply mask to missing values dimension 
            diff = pred[i, :] .- truth[i, :]
            masked_loss = sum(diff .^ 2 .* batch_mask) / (sum(batch_mask) + eps())
            total_loss += masked_loss
        else
            # Standard MSE for fully observed dimensions
            total_loss += Flux.mse(pred[i, :], truth[i, :])
        end
    end

    # Average the loss over dimensions
    avg_loss = total_loss / n_dims

    # Regularization
    params = [m.p.θ; m.p.θ1] 
    reg1 = λ1 * sum(abs, params)  # L1
    reg2 = λ2 * sum(abs2, params) # L2

    return avg_loss + reg1 + reg2
end

@doc """
    plot_model_performance(sol, t, X_sparse, train_70, model, U_re, U_truth, mask, dt)

Generates and returns four plots to evaluate model performance of an `n`-dimensional system:
1. **Trajectories Plot**: Compares predicted trajectories with ground truth for all dimensions (first 70 points).
2. **Interaction Terms Plot**: Compares neural network outputs with expected interaction terms (first 70 points).
3. **L2 Error Plot**: Computes and visualizes the total L2 error across all dimensions at each time step (first 70 points).
4. **Reconstructed Solution Plot**: Compares the full ground truth trajectory with NODE predictions over all time points.

# Arguments
- `sol`: Original solution object.
- `t`: The time vector associated with the original solution object.
- `X_sparse`: The sparsified ground truth.
- `train_70`: Training data batches of size 70. 
- `model`: The trained model used to predict system trajectories.
- `U_re`: A function that reconstructs interaction terms from the ANN parameters.
- `U_truth`: True interaction terms.
- `mask`: The global mask used to indicate available data points.
- `dt`: Time step interval between consecutive points.

# Returns
A tuple containing four plots:
1. `plt_traj`: The trajectories plot (first 70 points).
2. `plt_interaction`: The interaction terms plot (first 70 points).
3. `plt_l2_error`: The L2 error plot (first 70 points).
4. `plt_re`: The reconstructed solution plot (full time range).
"""
function plot_model_performance(sol, t, X_sparse, train_70, model, U_re, U_truth, mask, dt)
    # state data
    X = Array(X_sparse) # Ground truth
    n_dims = size(X, 1)

    # Extract last dimension's sparse values
    X_last_sparse = X[end, :]
    X_last_spskip = X_last_sparse[X_last_sparse .!= 0]  # Only nonzero values

    # Model predictions
    pred = model((t, X))

    # Trajectories Plot
    plt_traj = plot(t[1:70], pred[1, 1:70], label="UDE Approximation", lw=2, color=:red, legend=:topleft)
    for i in 2:n_dims
        plot!(plt_traj, t[1:70], pred[i, 1:70], label="", lw=2, color=:red)
    end
    scatter!(plt_traj, t[1:70], X[1, 1:70], label="Measurements", marker=:circle, color=:black)
    for i in 2:n_dims-1
        scatter!(plt_traj, t[1:70], X[i, 1:70], label="", marker=:circle, color=:black)
    end
    scatter!(plt_traj, t[1:70], X_last_spskip[1:70], label="", marker=:circle, color=:black)
    xlabel!(plt_traj, "t")
    ylabel!(plt_traj, "Population")

    # Interaction Terms Plot
    U_pred = U_re(model.p.θ)(X)
    plt_interaction = plot(t[1:70], U_pred[1, 1:70], label="UDE Approximation", lw=2, color=:red, legend=:topleft)
    for i in 2:n_dims
        plot!(plt_interaction, t[1:70], U_pred[i, 1:70], label="", lw=2, color=:red)
    end
    scatter!(plt_interaction, t[1:70], U_truth[1, 1:70], label="True Interaction", lw=2, color=:black)
    for i in 2:n_dims
        scatter!(plt_interaction, t[1:70], U_truth[i, 1:70], label="", lw=2, color=:black)
    end
    xlabel!(plt_interaction, "t")
    ylabel!(plt_interaction, "Interactions")

    # L2 Error Plot
    batch_t, batch_data = train_70[1] # First 70 points
    pred_70 = model((batch_t, batch_data))

    l2_error = zeros(length(batch_t))

    for j in 1:length(batch_t)
        diff = zeros(n_dims)
    
        for i in 1:n_dims
            if i == n_dims  # Apply mask only to the last dimension
                mask_batch = get_mask_for_batch(batch_t, mask, t[1], dt)
                diff[i] = (pred_70[i, j] * mask_batch[j] - batch_data[i, j] * mask_batch[j])
            else
                diff[i] = (pred_70[i, j] - batch_data[i, j])
            end
        end
    
        # Compute L2 norm over all dimensions for this time step
        l2_error[j] = norm(diff, 2)
    end

    # Prevent log(0) issues
    l2_error = max.(l2_error, eps())

    y_ticks = 10.0 .^ (-4:1.:2) # Custom ticks
    y_labels = ["10^{$(round(y, digits=1))}" for y in log10.(y_ticks)] # Format labels as 10^x

    plt_l2_error = plot(batch_t, l2_error, xlabel="t", ylabel="L2-Error", xlims=(0, batch_t[end]), yscale=:log10, yticks=(y_ticks, y_labels),  ylims=(minimum(y_ticks), maximum(y_ticks)), lw=2, color=:red, legend=false)

    # Reconstructed Solution Plot
    plt_re = plot(sol)
    plot!(plt_re, t, pred')

    return plt_traj, plt_interaction, plt_l2_error, plt_re
end

@doc """
    perform_symbolic_regression(X_sparse, dx, niterations=100; binary_operators=[+, *, -], unary_operators=[])

Perform symbolic regression to extract interpretable equations from a neural ODE model.

# Arguments
- `X_sparse`: Input features for regression.
- `dx`: The learned interaction terms from the neural ODE model.
- `niterations`: Number of iterations for the symbolic regression search (default=100).
- `binary_operators`: List of binary operators to use in equation search (default `[+, *, -]`).
- `unary_operators`: List of unary operators to use in equation search (default `[]`).

# Returns
- `hall_of_fame`: Best equations found for each dimension.
- `pareto_frontiers`: Pareto-optimal equations for each dimension.
"""
function perform_symbolic_regression(X_sparse, dx, niterations=100; binary_operators=[+, *, -], unary_operators=[])
    options = SymbolicRegression.Options(binary_operators=binary_operators, unary_operators=unary_operators)

    hall_of_fame = []
    pareto_frontiers = []
    n_dims = size(dx, 1)
    
    for i in 1:n_dims # Iterate over each dimension
        hof = EquationSearch(X_sparse, dx[i, :], niterations=niterations, options=options, parallelism=:multithreading)
        pareto = calculate_pareto_frontier(X_sparse, dx[i, :], hof, options)

        push!(hall_of_fame, hof)
        push!(pareto_frontiers, pareto)
    end

    return hall_of_fame, pareto_frontiers
end

