IF [%2]==[] (
	ECHO PORT NOT SPECIFIED USING DEFAULT PORT 44700
	"c:\Program Files\GGP SDK\dev\bin\chromeclient\chrome_client.exe" -gamelet-port 44700 -gamelet-ip %1 
	) ELSE ( 
	ECHO PORT IS SPECIFIED: %2
	"c:\Program Files\GGP SDK\dev\bin\chromeclient\chrome_client.exe" -gamelet-port %2 -gamelet-ip %1
	)
