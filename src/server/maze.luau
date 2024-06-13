local HttpService = game:GetService("HttpService")

local Maze = {}

function Maze.new(originPos, size, mazeString, mazeId)
    -- Variables    
    local self      = setmetatable({}, {__index = Maze})
    self.originPos  = originPos
    self.size       = size
    self.mazeString = mazeString
    self.mazeId     = mazeId
    self.mazeColor  = BrickColor.random()
    self.startPart  = nil
    self.exitPart   = nil
    return self
end

function Maze:setStartPart(part)
    self.startPart = part
end

function Maze:setExitPart(part)
    self.exitPart = part
end

function Maze:getStartPart()
    return self.startPart
end

function Maze:getExitPart()
    return self.exitPart
end

-- Fonction pour créer une part de mur
local function createWall(position, size, isPlayer, isExit, mazeFolder, mazeInstance)
    local part = Instance.new("Part")
    part.Size = size
    part.Position = position
    part.Anchored = true
    if isPlayer then
        part.BrickColor = BrickColor.new("Bright green")
        part.Transparency = 0.8
        part.CanCollide = false
        part.Name = "PlayerStart"
        mazeInstance:setStartPart(part)
    elseif isExit then
        part.BrickColor = BrickColor.new("Bright red")
        part.Name = "Exit"
        part.Transparency = 0.8
        part.CanCollide = false
        mazeInstance:setExitPart(part)
    else
        part.BrickColor = mazeInstance.mazeColor 
    end
    part.Material = Enum.Material.SmoothPlastic
    part.Parent = mazeFolder
end

-- Fonction pour créer le sol du labyrinthe
local function createFloor(size, mazeFolder, mazeInstance, numRows, numCols)
    local floor = Instance.new("Part")
    floor.Size = Vector3.new(size.X * numCols, 1, size.Z * numRows) -- Largeur et longueur relatives à la taille du maze
    floor.Position = mazeInstance.originPos + Vector3.new((size.X * numCols) / 2 - size.X / 2, -0.5, (size.Z * numRows) / 2 - size.Z / 2)
    floor.Anchored = true
    floor.BrickColor = BrickColor.new("Medium stone grey")
    floor.Material = Enum.Material.SmoothPlastic
    floor.Name = "MazeFloor"
    floor.Parent = mazeFolder
end

function Maze:generateMaze()
    local mazeFolder = Instance.new("Folder")
    mazeFolder.Name = "Maze_" .. self.mazeId
    mazeFolder.Parent = workspace

    -- Diviser la chaîne de caractères en lignes
    local mazeLines = {}
    for line in string.gmatch(self.mazeString, "[^\n]+") do
        table.insert(mazeLines, line)
    end

    -- Dimensions des parts et des cellules
    local partSize = Vector3.new(self.size, self.size, self.size)
    local halfPartSize = partSize / 2
    local numRows = #mazeLines
    local numCols = #mazeLines[1]

    -- Créer le sol du labyrinthe
    createFloor(partSize, mazeFolder, self, numRows, numCols)

    -- Parcourir chaque ligne et chaque caractère pour créer le labyrinthe
    for i, line in ipairs(mazeLines) do
        for j = 1, #line do
            local char = string.sub(line, j, j)
            local position = Vector3.new((j - 1) * partSize.X, halfPartSize.Y, (i - 1) * partSize.Z) + self.originPos
            if char == "#" then
                createWall(position, partSize, false, false, mazeFolder, self)
            elseif char == "P" then
                createWall(position, partSize, true, false, mazeFolder, self)
            elseif char == "E" then
                createWall(position, partSize, false, true, mazeFolder, self)
            end
        end
    end
end

return Maze
