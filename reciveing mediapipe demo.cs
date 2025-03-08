using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;

class SkeletonReceiver
{
    static void Main()
    {
        // Set up TCP listener for Python data
        TcpListener listener = new TcpListener(IPAddress.Parse("127.0.0.1"), 5005);
        listener.Start();
        Console.WriteLine("Listening for pose and gesture data...");

        while (true)
        {
            using (TcpClient client = listener.AcceptTcpClient())
            using (NetworkStream stream = client.GetStream())
            {
                byte[] buffer = new byte[4096];
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                string receivedData = Encoding.UTF8.GetString(buffer, 0, bytesRead);

                // Deserialize JSON from Python
                var skeletonData = JsonConvert.DeserializeObject<SkeletonData>(receivedData);

                Console.WriteLine("Received Data:");
                foreach (var landmark in skeletonData.Landmarks)
                {
                    Console.WriteLine($"{landmark.Key}: X={landmark.Value.X}, Y={landmark.Value.Y}");
                }

                if (!string.IsNullOrEmpty(skeletonData.Gesture))
                {
                    Console.WriteLine($"Detected Gesture: {skeletonData.Gesture}");
                }
            }
        }
    }
}

// Class for deserializing JSON data
class SkeletonData
{
    public Dictionary<string, Landmark> Landmarks { get; set; }
    public string Gesture { get; set; }
}

// Class for individual landmark points
class Landmark
{
    public float X { get; set; }
    public float Y { get; set; }
}
