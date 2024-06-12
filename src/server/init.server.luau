-- imports modulescript
local Roquest = require(script.roquest)
local Maze    = require(script.maze)
local Lidar   = require(script.lidar)
local Agent   = require(script.agent)


-- imports services
local HttpService = game:GetService("HttpService")


-- Variables


-- try catch
local success, errorMessage = pcall(function()

    local response    = Roquest:Request("GET", "http://169.254.58.42:8000/generate_maze")
	local mazeData    = HttpService:JSONDecode(response.Body)
	local mazeString  = mazeData.maze    
    local tmpMaze     = Maze.new(Vector3.new(0, 1, 0), 8, mazeString, 1)
    tmpMaze:generateMaze()
    local startPart   = tmpMaze:getStartPart()
    local tmpNemo     = Agent.new(tmpMaze)


    local response    = Roquest:Request("GET", "http://169.254.58.42:8000/generate_maze")
	local mazeData    = HttpService:JSONDecode(response.Body)
	local mazeString  = mazeData.maze    
    local tmpMaze2 = Maze.new(Vector3.new(0, 50, 0), 8, mazeString, 1)
    tmpMaze:generateMaze()
    local startPart2 = tmpMaze2:getStartPart()
    local tmpNemo2 = Agent.new(tmpMaze2)



    -- local response    = Roquest:Request("GET", "http://169.254.58.42:8000/generate_maze")
	-- local mazeData    = HttpService:JSONDecode(response.Body)
	-- local mazeString  = mazeData.maze    

    -- local tmpMaze = Maze.new(Vector3.new(0, 100, 0), 8, mazeString, 1)
    -- tmpMaze:generateMaze()

    -- local response    = Roquest:Request("GET", "http://169.254.58.42:8000/generate_maze")
	-- local mazeData    = HttpService:JSONDecode(response.Body)
	-- local mazeString  = mazeData.maze    

    -- local tmpMaze = Maze.new(Vector3.new(0, 150, 0), 8, mazeString, 1)
    -- tmpMaze:generateMaze()
end)


print(success, errorMessage)


if not success then
	warn("Serveur Python non démarré")
end
