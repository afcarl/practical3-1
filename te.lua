require 'mobdebug'.start()

local x = torch.rand(3,2)

local y = torch.rand(2,5)

local z = x

z:resize(6)
z[1] = 99

a = 1





