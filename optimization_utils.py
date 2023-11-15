import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------- #

def objective_function(X,y,lambda_val,w,problem_part=1):
  if problem_part == 1 :
    return objective_function_part_1(X,y,lambda_val,w)
  
  if problem_part == 2 :
    return objective_function_part_2(X,y,lambda_val,w)
  
# --------------------------- PART 1 ------------------------------------ #

def closed_form_solution(X,y,lambda_val):
  N = len(y)
  R_X = sum(np.outer(x,x) for x in X)
  identity_matrix = np.identity(X.shape[1])

  # w_optimal = np.linalg.inv(((1/N)*R_X) + (lambda_val*identity_matrix)) @ ((1/N)*X.T @ y)
  w_optimal = np.linalg.inv(((2/N)*R_X) + (lambda_val*identity_matrix)) @ ((2/N)*X.T @ y)
  # w_optimal = np.linalg.inv(R_X + lambda_val*identity_matrix) @ (X.T @ y)
  return w_optimal

def objective_function_part_1(X,y,w, lambda_val):
  return (1/len(y))*np.linalg.norm(np.dot(X,w) - y) + lambda_val*np.linalg.norm(w)**2

def gradient(R_x, xy, lambda_val, w, N):
  w = (2/N)*(np.dot(R_x,w) - xy + lambda_val*w)
  return w


def gradient_descent(X,y,lambda_val, lr, epochs, precision):
    fct_values = [] # to store the loss values
    N = len(y)
    d = X.shape[1]
    R_x = sum(np.outer(x,x) for x in X)
    xy = np.dot(X.T , y)

    # w = np.random.rand(d,1)
    w = np.random.rand(d)
    # print(f'[INFO] Initial weights values = {w}')

    # w = np.ones(d)
    # grad = gradient(R_x, xy, lambda_val, w, N)
    
    for epoch in tqdm(range(epochs)):

      grad = gradient(R_x, xy, lambda_val, w, N)
      w = w - lr * grad
      loss = objective_function_part_1(X,y,w, lambda_val)
      fct_values.append(loss)

      if epoch == 0:
        minimum_loss = loss
      else:
        if loss < minimum_loss:
          minimum_loss = loss

      # print(f'[INFO] grad value = {grad}')
      if (epoch > 0):
        delta_grads = np.linalg.norm(grad - prev_grad)
        if np.abs(delta_grads) < precision :
          print(f'[INFO] GD did converge at epoch {epoch}')
          return w, fct_values
        
      prev_grad = grad.copy() # store previous grad

    return w, fct_values, minimum_loss

def main_part_one(  dataset='CRIME', 
                    lambda_val = 1,
                    lr = 0.01,
                    epochs = 200000,
                    precision = 1e-32):
  
  if dataset.upper() == 'CRIME':
    # Load "Communities and Crime" dataset
    data = load_communities_and_crime_data()
    X, y = preprocess_data_crime(data) 
  elif dataset.upper() == 'ELECTRIC':
    # Load "Household electric power" dataset
    data = load_electric_power_data()
    X, y = preprocess_data_electric(data)   

  # Closed form solution
  print("Computing closed form solution...")
  w_optimal = closed_form_solution(X,y, lambda_val)
  print("Closed form solution :", w_optimal)

  # Gradient Descent
  w_GD, fct_values, minimum_loss = gradient_descent(X,y, lambda_val, lr, epochs, precision)
  print("Gradient Descent Solution :", w_GD)

  # Comparison of closed form and GD
  # f_w_GD = objective_function_part_1(X,y,w_GD, lambda_val)
  f_w_opti = objective_function_part_1(X,y,w_optimal, lambda_val)
  print("f(w*)=", f_w_opti," f(w_GD)=", minimum_loss," f(w_GD)-f(w*)=", minimum_loss - f_w_opti)

  # Plot f(wk) for GD
  plt.plot(fct_values, label='GD')
  plt.xlabel('Epochs')
  plt.ylabel('f(w_k)')
  plt.legend()
  plt.title(f'Convergence of f(w_k) for GD - {dataset.upper()} Dataset')
  plt.show()

# --------------------------- PART 2 ------------------------------------ #
def objective_function_part_2(X,y,lambda_val,w, indices):
  X_used = X[indices]
  y_used = y[indices]
  N = len(y_used)
  # if np.any(np.dot(-1*y.T,np.dot(X,w))) >= 700: #to avoid the divergence of exponential in numpy
  x_times_w = np.dot(X_used, w)
  result = -1*np.dot(y_used.T, x_times_w)
  exp = np.exp(result)

  # if (result > 700).all():
  #   exp = np.exp(700)
  # else :
  #   exp = np.exp(np.dot(-1*y.T,np.dot(X,w)))
  f_w = (1/N)*np.log(1+exp) + lambda_val*np.linalg.norm(w)**2
  # print(f'objective_function_part_2 : X_used.shape = {X_used.shape}')
  # print(f'objective_function_part_2 : y_used.shape = {y_used.shape}')
  # print(f'objective_function_part_2 : w.shape = {w.shape}')
  # print(f'objective_function_part_2 : x_times_w.shape = {x_times_w.shape}')
  # print(f'objective_function_part_2 : x_times_w.shape = {x_times_w.shape}')
  # print(f'objective_function_part_2 : result = {result}')
  # print(f'objective_function_part_2 : exp = {exp}')
  # print(f'objective_function_part_2 : f_w = {f_w}')
  return f_w

def gradient_part_2(X,y,lambda_val,w,indices):
  X_used = X[indices]
  y_used = y[indices]
  N = len(indices)
  result = np.dot(-1 * y.T, np.dot(X, w))
  exp = np.exp(result)
  y_times_x = np.dot(X_used.T,y_used)
  grad = (-1/N)*(y_times_x * exp)/(1+exp) + 2*lambda_val*w
  return grad


def stochastic_gradient_descent(X,y,lambda_val,lr,epochs,precision, gradient_method='plain'):
    fct_values = [] # to store the loss values
    N = len(y)
    d = X.shape[1]
    w = np.random.rand(d)

    for epoch in tqdm(range(epochs)):
      
      if gradient_method.upper() == 'STOCHASTIC':
        x_index = [np.random.randint(1,N)]
      else:
        x_index = [i for i in range(N)]

      grad = gradient_part_2(X,y,lambda_val,w,x_index)
      w = w - lr * grad
      loss = objective_function_part_2(X,y,lambda_val, w, x_index)
      fct_values.append(loss)
      # print(f'[INFO] grad value = {grad}')

      if epoch == 0:
        minimum_loss = loss
      else:
        if loss < minimum_loss:
          minimum_loss = loss

      if (epoch > 0):
        delta_grads = np.linalg.norm(grad - prev_grad)
        if np.abs(delta_grads) < precision :
          print(f'[INFO] GD did converge at epoch {epoch}')
          return w, fct_values
        
      prev_grad = grad.copy() # store previous grad

    return w, fct_values, minimum_loss


def main_part_2(  dataset='CRIME', 
                  lambda_val = 1,
                  lr = 0.01,
                  epochs = 200000,
                  precision = 1e-32):
  
  if dataset.upper() == 'CRIME':
    # Load "Communities and Crime" dataset
    data = load_communities_and_crime_data()
    X, y = preprocess_data_crime(data) 
  elif dataset.upper() == 'ELECTRIC':
    # Load "Household electric power" dataset
    data = load_electric_power_data()
    X, y = preprocess_data_electric(data)   

  # Gradient Descent
  w_GD, fct_values_GD, minimum_loss_GD = stochastic_gradient_descent(X,y, lambda_val, lr, epochs, precision, gradient_method='plain')
  print("Gradient Descent Solution :", w_GD)

  # Stochastic Gradient Descent
  w_SGD, fct_values_SGD, minimum_loss_SGD = stochastic_gradient_descent(X,y, lambda_val, lr, epochs, precision, gradient_method='stochastic')
  print("Stochastic Gradient Descent Solution :", w_SGD)

  # Comparison
  print("Part II :", " f(w_GD)=", minimum_loss_GD," f(w_SGD)=", minimum_loss_SGD," f(w_GD)-f(w_SGD)=", minimum_loss_GD - minimum_loss_SGD)

  # Plot f(wk) for GD
  plt.plot(fct_values_GD, label='GD')
  plt.plot(fct_values_SGD, label='SGD')
  plt.xlabel('epoch')
  plt.ylabel('f(w_k)')
  plt.legend()
  plt.title(f'Convergence of f(w_k) for Both gradient methods - {dataset.upper()} Dataset')
  plt.show()