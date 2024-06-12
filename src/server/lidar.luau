-- Définition de la classe Lidar
local Lidar = {}
Lidar.__index = Lidar

function Lidar.new(agent, numRays, maxDistance, originOffset)
    local self = setmetatable({}, Lidar)
    -- Variables
    self.agent = agent
    self.numRays = numRays or 360
    self.maxDistance = maxDistance or 100
    self.originOffset = originOffset or 0.4
    self.lidarParts = {} -- Table pour stocker les parts générés
    self.head = agent:WaitForChild("Head")
    
    self:initializeLidarParts()
    return self
end

function Lidar:initializeLidarParts()
    for i = 1, self.numRays do
        local rayPart = Instance.new("Part")
        rayPart.Anchored = true
        rayPart.CanCollide = false
        rayPart.Size = Vector3.new(0.2, 0.2, self.maxDistance)
        rayPart.Color = Color3.new(0, 1, 0) -- Vert par défaut
        rayPart.Parent = workspace
        rayPart.Transparency = 0.8
        table.insert(self.lidarParts, {part = rayPart, isObstacle = false})
    end
end

function Lidar:interpolateColor(startColor, endColor, alpha)
    return startColor:Lerp(endColor, alpha)
end

function Lidar:performLidarScan()
    local results = {}
    local angleIncrement = 360 / self.numRays

    for i = 0, self.numRays - 1 do
        local angle = math.rad(i * angleIncrement)
        local direction = self.head.CFrame:VectorToWorldSpace(Vector3.new(math.cos(angle), 0, math.sin(angle)))
        local rayOrigin = self.head.Position + direction * self.originOffset
        local rayDirection = direction * self.maxDistance
        local raycastParams = RaycastParams.new()
        raycastParams.FilterDescendantsInstances = {self.head.Parent}
        raycastParams.FilterType = Enum.RaycastFilterType.Blacklist

        for _, partData in ipairs(self.lidarParts) do
            table.insert(raycastParams.FilterDescendantsInstances, partData.part)
        end

        local raycastResult = workspace:Raycast(rayOrigin, rayDirection, raycastParams)
        local partData = self.lidarParts[i + 1]
        local rayPart = partData.part

        if raycastResult then
            local distance = raycastResult.Distance
            local alpha = distance / self.maxDistance
            local color = self:interpolateColor(Color3.new(1, 0, 0), Color3.new(0, 1, 0), alpha)

            rayPart.Color = color
            rayPart.Size = Vector3.new(0.2, 0.2, distance)
            rayPart.CFrame = CFrame.new(rayOrigin, raycastResult.Position) * CFrame.new(0, 0, -distance / 2)
            table.insert(results, {
                Position = raycastResult.Position,
                Distance = distance,
                Part = raycastResult.Instance
            })
        else
            rayPart.Color = Color3.new(0, 1, 0) -- Vert pour les rayons qui ne rencontrent pas d'obstacle
            rayPart.Size = Vector3.new(0.2, 0.2, self.maxDistance)
            rayPart.CFrame = CFrame.new(rayOrigin, rayOrigin + rayDirection) * CFrame.new(0, 0, -self.maxDistance / 2)
            table.insert(results, {
                Position = rayOrigin + rayDirection,
                Distance = self.maxDistance,
                Part = nil
            })
        end
    end

    return results
end

function Lidar:moveRig(results)
    local distances = {}

    for i, result in ipairs(results) do
        table.insert(distances, {result = result, distance = result.Distance})
    end

    table.sort(distances, function(a, b) return a.distance > b.distance end)

    local bestDirection = distances[1]
    if bestDirection then
        local index = table.find(results, bestDirection.result)
        local angle = math.rad((index - 1) * (360 / self.numRays))
        local direction = self.head.CFrame:VectorToWorldSpace(Vector3.new(math.cos(angle), 0, math.sin(angle)))

        self.agent.Humanoid:Move(direction, false)
    end
end

function Lidar:createPathSphere(position)
    local pathSphere = Instance.new("Part")
    pathSphere.BrickColor = BrickColor.new("Really red")
    pathSphere.CanCollide = false
    pathSphere.Transparency = 0.5
    pathSphere.Size = Vector3.new(1, 1, 1)
    pathSphere.Anchored = true
    pathSphere.Position = position
    pathSphere.Shape = Enum.PartType.Ball
    pathSphere.Material = Enum.Material.SmoothPlastic
    pathSphere.Parent = game.Workspace.Path
end

function Lidar:recordPath(path, position)
    local gridX = math.floor(position.X / 8)
    local gridY = math.floor(position.Y / 8)
    local gridZ = math.floor(position.Z / 8)

    table.insert(path, {X = gridX, Y = gridY, Z = gridZ})
end

function Lidar:start()
    coroutine.wrap(function()
        local path = {}
        local cpt = 0

        while true do
            local results = self:performLidarScan()
            self:moveRig(results)
            task.wait(0.1)

            self:recordPath(path, self.agent.HumanoidRootPart.Position)

            cpt = cpt + 1
            if cpt % 2 == 0 then
                self:createPathSphere(self.agent.HumanoidRootPart.Position)
            end
        end
    end)()
end

-- Configuration
return Lidar
