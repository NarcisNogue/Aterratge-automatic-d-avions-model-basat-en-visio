using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
 
using System.Net;  
using System.Net.Sockets;  
using System.Text;  


public class plane_control : MonoBehaviour
{
    // Public variables
    public float thrustForce;
    Rigidbody body;

    // Private variables
    private float baseCl = 1f;
    private float maxThrust = 70f;
    private float thrust_step = 10f;
    private Vector3 forward = new Vector3(1, 0, 0);
    private Vector3 up = new Vector3(0, 1, 0);
    private Vector3 right = new Vector3(0, 0, 1);

    private float elevon_pos;
    private float max_elevon_pos = 20f;

    GameObject horizontal_stab;
    GameObject elevon;

    public Camera cam1;
    public Camera cam2;

    public RenderTexture cam1out;
    static IPAddress ipAddress = IPAddress.Parse("127.0.0.1");
    IPEndPoint remoteEP = new IPEndPoint(ipAddress,65432);

    Socket socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp );

    byte[] bytes = new byte[1024];

    byte[] msgEof = Encoding.ASCII.GetBytes("<EOF>");

    private byte[] rawByteData;
    private Texture2D texOut;
    
    void Awake () {
        QualitySettings.vSyncCount = 0;  // VSync must be disabled
        Application.targetFrameRate = 30;
    }

    void OnDestroy () {
        byte[] msgEnd = Encoding.ASCII.GetBytes("END_NOW<EOF>");
        socket.Send(msgEnd);
    }


    // Start is called before the first frame update
    void Start()
    {
        socket.Connect(remoteEP);
        thrustForce = maxThrust / 2f;
        body = GetComponent<Rigidbody>();
        body.velocity = transform.TransformDirection(new Vector3(15, 0, 0));
        elevon_pos = 0f;

        cam1.enabled = true;
        cam2.enabled = false;


        // Vars for sending
        rawByteData = new byte[cam1out.width * cam1out.height * 3];
        texOut = new Texture2D(cam1out.width, cam1out.height, TextureFormat.RGB24, false);

    }

    // Update is called once per frame
    void Update()
    {

        // Variables locals
        

        // Calculs inicials
        var velocity = GetComponent<Rigidbody>().velocity;
        var localVel = transform.InverseTransformDirection(velocity);
        var angle_attack = -ToDegrees(Math.Atan2(localVel.y, localVel.x));
        var horizontal_angle = ToDegrees(Math.Atan2(localVel.z, localVel.x));
        var angularVelocity = body.angularVelocity;
        float yTorque = - angularVelocity.x * 3f;

        var elevon_speed = 0.2f;

        // var dragRotation = Quaternion.FromToRotation(transform.TransformDirection(forward), velocity);

        // ------- INPUTS ---------------
        if (Input.GetKeyDown("q")){
            byte[] msgEndAll = Encoding.ASCII.GetBytes("END_PROGRAM<EOF>");
            socket.Send(msgEndAll);
        }
        if (Input.GetKeyDown("z")){
            thrustForce = thrustForce + thrust_step <= maxThrust ? thrustForce + thrust_step : maxThrust;
        }
        if (Input.GetKeyDown("x")){
            thrustForce = thrustForce - thrust_step >= 0f ? thrustForce - thrust_step : 0f;
        } 
        
        if (Input.GetKey(KeyCode.DownArrow)){
            elevon_pos = elevon_pos - elevon_speed > - max_elevon_pos ? elevon_pos - elevon_speed : -max_elevon_pos;
        } else if( elevon_pos < 0f) {
            elevon_pos += elevon_speed;
        }
        if (Input.GetKey(KeyCode.UpArrow)){
            elevon_pos = elevon_pos + elevon_speed < + max_elevon_pos ? elevon_pos + elevon_speed : max_elevon_pos;
        } else if( elevon_pos > 0f) {
            elevon_pos -= elevon_speed;
        }

        var elevon_angle = angle_attack - 2 + elevon_pos;

        if (Input.GetKey(KeyCode.LeftArrow)){
            yTorque = 80f;
        }
        if (Input.GetKey(KeyCode.RightArrow)){
            yTorque = -80f;
        }

        if(Input.GetKey(KeyCode.RightControl)){
            horizontal_angle += 15f;
        } else if(Input.GetKey(KeyCode.Keypad0)){
            horizontal_angle -= 15f;
        }

        if (Input.GetKeyDown(KeyCode.C)) {
            cam1.enabled = !cam1.enabled;
            cam2.enabled = !cam2.enabled;
        }


        var Cl = baseCl * Math.Sin(ToRadians(2*angle_attack));
        var elevonCl = baseCl * Math.Sin(ToRadians(2*elevon_angle));
        var aleronCl = baseCl * Math.Sin(ToRadians(2*horizontal_angle));
        // var Cd = baseCd * (1 - Math.Cos(ToRadians(2*angle_attack)));

        var area = 2.5f;
        var elevon_area = 1.8f;
        var aleron_area = .5f;

        var lift = Cl * area * .5f * Math.Pow(localVel.x, 2);
        var elevon_lift = elevonCl * elevon_area * .5f * Math.Pow(localVel.x, 2);
        var aleron_lift = aleronCl * aleron_area * .5f * Math.Pow(localVel.x, 2);
        // var drag = Cd * area * .5f * Math.Pow(localVel.x, 2);

        // FORCES
        body.AddRelativeForce( forward * thrustForce * 2.5f );
        // body.AddRelativeForce(-forward * Convert.ToSingle(drag) );
        body.AddRelativeForce( up * Convert.ToSingle(lift) );

        // TORQUES
        body.AddRelativeTorque ( - right * Convert.ToSingle(elevon_lift) );
        body.AddRelativeTorque ( - up * Convert.ToSingle(aleron_lift) );
        body.AddRelativeTorque ( forward * yTorque );
        // body.AddTorque(dragRotation * 0.01 * Math.Pow(localVel.x, 2));

        // SEND RENDER ---------------------
        RenderTexture.active = cam1out;
        texOut.ReadPixels(new Rect(0, 0, cam1out.width, cam1out.height), 0, 0);
        texOut.Apply();
        // Array.Copy(, rawByteData, rawByteData.Length);
        socket.Send(texOut.EncodeToJPG());

        
        socket.Send(msgEof);

        // int bytesRec = socket.Receive(bytes);
        // Debug.Log(Encoding.ASCII.GetString(bytes,0,bytesRec)); 
        // DEBUG

        Debug.Log((1.0)/Time.deltaTime);
    }

    // ######### PRIVATE METHODS ###################################

    private double ToRadians(double angle)
    {
        return (Math.PI / 180) * angle;
    }

    private double ToDegrees(double angle)
    {
        return (180 / Math.PI) * angle;
    }
}
