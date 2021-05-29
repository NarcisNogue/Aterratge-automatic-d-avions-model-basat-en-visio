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
    private float minThrust = 0f;
    private float thrust_step = 10f;
    private Vector3 forward = new Vector3(1, 0, 0);
    private Vector3 up = new Vector3(0, 1, 0);
    private Vector3 right = new Vector3(0, 0, 1);

    private Vector2 targetPoint;
    private Vector2 targetAngle = new Vector2(0, 0);
    private List<Vector2> pointHistory = new List<Vector2>();
    private Vector2 target = new Vector2(0, 20); // Angles respecte el centre de la imatge en una camera de 60º FOV
    // Corresponent al punt 64, 40 aprox. en una imatge de 128x128
    private float elevon_pos;
    private float max_elevon_pos = 20f;
    private float max_yTorque = 80;

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


    // PID VARS -------------
    //Roll
    private float RollP1;
    private float RollI1;
    private float RollD1;
    private float prevRoll1;
    private float RollSum1;
    private float RollP2;
    private float RollI2;
    private float RollD2;
    private float prevRoll2;
    private float RollSum2;
    //Pitch
    private float PitchP1;
    private float PitchI1;
    private float PitchD1;
    private float prevPitch1;
    private float PitchSum1;
    private float PitchP2;
    private float PitchI2;
    private float PitchD2;
    private float prevPitch2;
    private float PitchSum2;

    private bool manual_control;
    private double max_roll_angle;
    private double max_pitch_angle;

    RaycastHit hit;
    float altitude = 0f;


    private float rollOnGetFloor = 0f;
    private float pitchOnGetFloor = 0f;
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

        // Set variables
        targetPoint = new Vector2(64, 40);

        manual_control = false;


        // Vars for sending
        rawByteData = new byte[cam1out.width * cam1out.height * 3];
        texOut = new Texture2D(cam1out.width, cam1out.height, TextureFormat.RGB24, false);


        // PIDs Setups
        RollP1 = 1f;
        RollI1 = 0.000f;
        RollD1 = 0.05f;
        RollP2 = 10f;
        RollI2 = 0.000f;
        RollD2 = 0.5f;
        prevRoll1 = 0f;
        prevRoll2 = 0f;
        RollSum1 = 0f;
        RollSum2 = 0f;

        //Pitch ----
        PitchP1 = 0.05f;
        PitchI1 = 0.01f;
        PitchD1 = 0.1f;
        PitchP2 = 0.1f;
        PitchI2 = 0.0005f;
        PitchD2 = 0.5f;
        prevPitch1 = 0f;
        prevPitch2 = 0f;
        PitchSum1 = 0f;
        PitchSum2 = 0f;


        max_roll_angle = 20.0;
        max_pitch_angle = 20.0;

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

        var elevon_angle = 0.0;
        if(manual_control) {
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

            elevon_angle = angle_attack - 2 + elevon_pos;

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
        }

        if (Input.GetKeyDown(KeyCode.C)) {
            cam1.enabled = !cam1.enabled;
            cam2.enabled = !cam2.enabled;
        }

        if (Input.GetKeyDown(KeyCode.M)) {
            manual_control = !manual_control;
        }


        // AUTOPILOT -----------------------------------------
        // Get altitude        
        if (Physics.Raycast(body.transform.position, Vector3.down , out hit, Mathf.Infinity))
        {
            if(altitude == Mathf.Infinity && hit.distance < 50){
                // Acabem de trobar el terra
                altitude = hit.distance;
                rollOnGetFloor = (float)ToDegrees(body.transform.rotation.x);
                pitchOnGetFloor = (float)ToDegrees(body.transform.rotation.z);
            } else if(hit.distance < 50) {
                altitude = hit.distance;
            } else {
                altitude = Mathf.Infinity;
            }
        }



        if(!manual_control){
            // Roll
            var diffX = targetAngle[0] - body.transform.rotation.y;
            //Sigmoid for P
            var P = (max_roll_angle*2)/(1 + Math.Exp(-diffX/max_roll_angle)) - max_roll_angle;
            var RollTarget =  -P*RollP1 - RollD1*(diffX - prevRoll1) - RollI1 * RollSum1;
            prevRoll1 = diffX;
            RollSum1 += diffX;
            RollSum1 *= 0.99f;

            if(altitude < 10){
                RollTarget = 0f;
            }

            var rollError = RollTarget - ToDegrees(body.transform.rotation.x);

            var AleronTarget = -RollP2*rollError - RollD2*(rollError - prevRoll2) - RollI2 * RollSum2;
            prevRoll2 = (float)rollError;
            RollSum2 += (float)rollError;
            RollSum2 *= 0.99f;

            yTorque = - Math.Min(Math.Max((float)AleronTarget, -max_yTorque), max_yTorque);
            


            // Pitch and thrust (es controlen alhora, +Pitch -> +Thrust)

            var diffY = targetAngle[1] - body.transform.rotation.z;

            var P2 = (max_pitch_angle*2)/(1 + Math.Exp(-diffX)) - max_pitch_angle;
            var PitchTarget =  -P2*PitchP1 - PitchD1*(diffY - prevPitch1) - PitchI1 * PitchSum1;
            prevPitch1 = diffY;
            PitchSum1 += diffY;
            PitchSum1 *= 0.99f;

            if(altitude < 15){
                PitchTarget = 0f;
            }

            var pitchError = PitchTarget - ToDegrees(body.transform.rotation.z);

            var AileronTarget = -PitchP2*pitchError - PitchD2*(pitchError - prevPitch2) - PitchI2 * PitchSum2;
            prevPitch2 = (float)pitchError;
            PitchSum2 += (float)pitchError;
            PitchSum2 *= 0.99f;

            Debug.Log(
                diffY.ToString() + ", "
                + PitchTarget.ToString() + ", " 
                + ToDegrees(body.transform.rotation.z).ToString() 
                + ", " + pitchError.ToString()
                + ", " + AileronTarget.ToString()             
            );
            
            elevon_angle = Math.Min(Math.Max((float)AileronTarget, -max_elevon_pos), max_elevon_pos);
            thrustForce = ((float)PitchTarget + 10f) / 10f / 2 * maxThrust;
            thrustForce = Math.Min(Math.Max(thrustForce, minThrust), maxThrust);

            if(altitude < 15) {
                thrustForce = 5f;
            }
            if(altitude < 3) {
                thrustForce = 0f;
            }
            
            // Temporal
            // elevon_angle = 0f;
        }

        Debug.Log(altitude);


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


        // MOVE CONTROL SURFACES (Nomes estetic)
        // var targetAngle = new Vector3(elevon_angle, 0, 0);
        // var angleActual = elevon.transform.rotation.eulerAngles;
        // for(int i = 0; i < 3; i++)
        //     angleActual[i] -= (angleActual[i] >= 180) ? 360 : 0;
        // var eulerAngleVelocity = targetAngle - angleActual;
        // //Debug.Log(angleActual);
        // //m_Rigidbody.MoveRotation(Quaternion.Lerp(rotationA, rotationB, Time.deltaTime/*ratio/mov_time*/));
        // var deltaRotation = Quaternion.Euler(eulerAngleVelocity * velocity * Time.deltaTime);
        // elevon.m_RigidBody.MoveRotation(elevon.m_RigidBody.rotation * deltaRotation);

        // SEND RENDER ---------------------
        RenderTexture.active = cam1out;
        texOut.ReadPixels(new Rect(0, 0, cam1out.width, cam1out.height), 0, 0);
        texOut.Apply();
        // Array.Copy(, rawByteData, rawByteData.Length);
        socket.Send(texOut.EncodeToJPG());

        
        socket.Send(msgEof);

        int bytesRec = socket.Receive(bytes);
        string response = Encoding.ASCII.GetString(bytes,0,bytesRec);
        if(response != "None") {
            targetPoint[0] = float.Parse(response.Split(',')[0]);
            targetPoint[1] = float.Parse(response.Split(',')[1]);
            pointHistory.Add(targetPoint);
            if(pointHistory.Count > 10) pointHistory.RemoveAt(0);

            var meanX = 0f;
            var meanY = 0f;
            foreach(var point in pointHistory) {
                meanX += point[0];
                meanY += point[1];
            }

            // Get target angle from point in image -----
            // Local angle
            var angle_loc_x = (meanX / pointHistory.Count - 64)/64*30;
            var angle_loc_y = (meanY / pointHistory.Count - 64)/64*30;

            targetAngle[0] = body.transform.rotation.y + angle_loc_x;
            targetAngle[1] = body.transform.rotation.z + angle_loc_y;

        }
        // Debug.Log(ToDegrees(RollTarget).ToString() + "," + ToDegrees(body.transform.rotation.x).ToString());
        // DEBUG

        // Debug.Log((1.0)/Time.deltaTime);
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
