-- imports modulescript
local Roquest = require(script.roquest)
local Maze    = require(script.maze)
local Lidar   = require(script.lidar)
local Agent   = require(script.agent)


-- imports services
local HttpService = game:GetService("HttpService")


-- Variables
local endPointIP   = "127.0.0.1"
local mazeSize     = 3 -- <
local wallMazeSize = 8
local nbrMaze      = 10
local cptCol       = 0
local cptRow       = 0
local mazes        = {}
local agents       = {}

-- try catch
local success, errorMessage = pcall(function()

    local mazesPerRow = math.sqrt(nbrMaze) -- Nombre de labyrinthes par ligne pour former une grille

    for i = 1, nbrMaze do
        local response   = Roquest:Request("GET", "http://" .. endPointIP .. ":8000/generate_maze")
        local mazeData   = HttpService:JSONDecode(response.Body)
        local mazeString = mazeData.maze
        local tmpMaze    = Maze.new(Vector3.new(cptCol, 1, cptRow), wallMazeSize, mazeString, 1)
        tmpMaze:generateMaze()
        --local startPart   = tmpMaze:getStartPart()
        --local tmpNemo    = Agent.new(tmpMaze, 1)
        table.insert(mazes, tmpMaze)

        cptCol = cptCol + (wallMazeSize * mazeSize) + wallMazeSize*2  *  mazeSize / 2 + wallMazeSize
    
        if (i % mazesPerRow) == 0 then
            cptCol = 0
            cptRow = cptRow + (wallMazeSize * mazeSize) + wallMazeSize*2 * mazeSize / 2 + wallMazeSize
        end
        task.wait()
    end



    for i = 1, nbrMaze do
        local tmpNemo = Agent.new(mazes[i], 1)
        table.insert(agents, tmpNemo)
        task.wait()
    end

    -- local response    = Roquest:Request("GET", "http://" .. endPointIP .. ":8000/generate_maze")
	-- local mazeData    = HttpService:JSONDecode(response.Body)
	-- local mazeString  = mazeData.maze    
    -- local tmpMaze     = Maze.new(Vector3.new(0, 1, 0), 8, mazeString, 1)
    -- tmpMaze:generateMaze()
    -- local startPart   = tmpMaze:getStartPart()
    -- local tmpNemo     = Agent.new(tmpMaze, 1)


    -- local response    = Roquest:Request("GET", "http://" .. endPointIP .. ":8000/generate_maze")
	-- local mazeData    = HttpService:JSONDecode(response.Body)
	-- local mazeString  = mazeData.maze    
    -- local tmpMaze2    = Maze.new(Vector3.new(0, 50, 0), 8, mazeString, 1)
    -- tmpMaze2:generateMaze()
    -- local startPart2  = tmpMaze2:getStartPart()
    -- local tmpNemo2    = Agent.new(tmpMaze2, 1)



end)


print(success, errorMessage)


if not success then
	warn("Serveur Python non démarré")
end
