local ReplicatedStorage   = game:GetService("ReplicatedStorage")
local Nemo                = ReplicatedStorage:WaitForChild("Assets"):WaitForChild("Nemo")
local ServerScriptService = game:GetService("ServerScriptService")
local Lidar               = require(ServerScriptService:WaitForChild("Server"):WaitForChild("lidar"))
local Maze                = require(ServerScriptService:WaitForChild("Server"):WaitForChild("maze"))

local function teleportRigIntoMaze(agent, maze)
    local startPart = maze:getStartPart()
    local rigSize   = agent.PrimaryPart.Size
    local mazePos   = startPart.Position
    local targetPos = mazePos + Vector3.new(0, rigSize.Y/2+1, 0)

    agent:SetPrimaryPartCFrame(CFrame.new(targetPos))
end

local Agent = {}

function Agent.new(maze, speed)
    local self = setmetatable({}, {__index = Agent})

    -- Variables
    local agent       = Nemo:Clone()
    self.agent        = agent
    self.Lidar        = Lidar.new(self.agent, 8, 10, 0.4)
    self.agent.Parent = game.Workspace:WaitForChild("Agents")
    self.maze         = maze

    self.agent.Head.BrickColor = BrickColor.random()

    -- Set speed
    self.agent.Humanoid.WalkSpeed = speed


    teleportRigIntoMaze(self.agent, maze)

    -- start lidar
    self.Lidar:start()

    return self
end

return Agent
