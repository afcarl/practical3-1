---------------------------------------------------------------------------------------
-- Practical 3 - Learning to use different optimizers with logistic regression
--
-- to run: th -i practical3.lua
-- or:     luajit -i practical3.lua
---------------------------------------------------------------------------------------
require 'mobdebug'.start()
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'




local p = torch.Tensor({{0,0,-1,0,-1,1,1},
{-1,1,1,-1,0,1,1},
{0,1,1,0,0,-1,1},
{-1,1,1,0,0,1,1},
{0,1,1,0,0,1,1},
{1,-1,1,1,1,-1,0},
{-1,1,-1,0,-1,0,1},
{0,-1,0,1,1,-1,-1},
{0,0,-1,1,1,0,-1}})

local lambda = 0.1
local c = torch.abs(p)
local n = 2
local parameters = torch.rand(n * (p:size(1) + p:size(2)))


local function feval(parameters)
    local x = parameters[{{1, p:size(1) * n}}]
    x:resize(p:size(1), n)
    local y = parameters[{{p:size(1) * n+1, parameters:size(1)}}]
    y:resize(n, p:size(2))
    local loss = torch.cmul(c, (torch.pow((p-x*y),2))):sum() + lambda * torch.pow(x,2):sum() + lambda * torch.pow(y,2):sum()
    local dloss_dx = torch.zeros(x:size(1), x:size(2))
    local dloss_dy = torch.zeros(y:size(1), y:size(2))
    
    for u = 1, x:size(1) do 
      local x_u = x[{{u}, {}}]
      local p_u = p[{{u},{}}]
      local p_u_resized = p_u:clone():resize(p_u:size(2))
      local c_u = torch.diag(torch.abs(p_u_resized))
      local dloss_dx_u = y*c_u*p_u:t() * (-2) + y * c_u * y:t()*x_u:t() * 2 + x_u * 2 * lambda
      dloss_dx[{{u}, {}}] = dloss_dx_u
    end
    
    for i = 1, y:size(2) do 
      local y_i = y[{{}, {i}}]
      local p_i = p[{{},{i}}]
      local p_i_resized = p_i:clone():resize(p_i:size(1))
      local c_i = torch.diag(torch.abs(p_i_resized))
      local dloss_dy_i = x:t() * c_i:t() * p_i * (-2) + x:t() * c_i * x * y_i * 2 + y_i * 2 * lambda
      dloss_dy[{{}, {i}}] = dloss_dy_i
    end
    
    local grad_parameters = torch.zeros(n * (p:size(1) + p:size(2)))
    dloss_dx:resize(dloss_dx:size(1) * dloss_dx:size(2))
    dloss_dy:resize(dloss_dy:size(1) * dloss_dy:size(2))
    grad_parameters[{{1, dloss_dx:size(1)}}] = dloss_dx
    grad_parameters[{{dloss_dx:size(1)+1, grad_parameters:size(1)}}] = dloss_dy
    
    return loss, grad_parameters
    
end


state = {
    learningRate = 1e-1,
    weightDecay = 1e-5,
    momentum = 1e-1,
    learningRateDecay = 1e-7
  }


for iterations = 1, 20 do 
  
  optim.sgd(feval, parameters, state)
  
  
end

local x = parameters[{{1, p:size(1) * n}}]
x:resize(p:size(1), n)
local y = parameters[{{p:size(1) * n+1, parameters:size(1)}}]
y:resize(n, p:size(2))
local p_predicted = x*y

print(p_predicted)
print(p)


