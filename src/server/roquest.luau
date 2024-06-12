local roquest = {}

local HttpService = game:GetService("HttpService")


function roquest:Request(method, url, headers, body)
    local response = HttpService:RequestAsync({
        Url = url,
        Method = method,
        Headers = headers,
        Body = body
    })

    return response
end

return roquest