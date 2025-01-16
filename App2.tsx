import { CameraView, CameraType, useCameraPermissions, Camera } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { captureRef } from "react-native-view-shot";
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';


export default function App() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const ws = useRef<WebSocket | null>(null); // Persist WebSocket instance
  const cameraRef = useRef(null);// Reference for CameraView

  const WEBSOCKET_URL = 'ws://192.168.66.26:8080'; // Replace with your Raspberry Pi's IP

  // Initialize WebSocket
  useEffect(() => {
    ws.current = new WebSocket(WEBSOCKET_URL);

    ws.current.onopen = () => {
      console.log("WebSocket connection opened");
    };

    ws.current.onmessage = (event) => {
      console.log("Received message from server:", event.data);
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.current.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => {
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, []);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }
  const saveToCameraRoll = async (fileUri: string) => {
    const permission = await MediaLibrary.requestPermissionsAsync();
    if (permission.granted) {
      await MediaLibrary.saveToLibraryAsync(fileUri);
      alert("Image saved to Photos!");
    } else {
      alert("Permission to access Photos is required.");
    }
  };
  const onSaveImageAsync = async () => {
    try {
      // const localUri = await captureRef(cameraViewRef, {
      //   format: 'jpg',
      //   quality: 1,
      // });
      const localUri = await cameraRef.current.takePictureAsync();
      await saveToCameraRoll(localUri.uri);
      //const hi = await cameraRef.current.CameraCapturedPicture.base64;
      console.log(localUri.uri);

      if (localUri) {
        const base64String = await FileSystem.readAsStringAsync(localUri.uri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        console.log(base64String);

        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(base64String);
          console.log("Screenshot sent successfully!");
        } else {
          console.error("WebSocket is not open. Current state:", ws.current?.readyState);
        }
      }
    } catch (error) {
      console.error("Error capturing and sending image:", error);
    }
  };

  const toggleCameraFacing = () => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  };

  return (
    <View style={styles.container}>
      <View style={styles.container} collapsable={false}>
        <CameraView style={styles.camera} facing={facing} ref = {cameraRef}>
          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.button} onPress={onSaveImageAsync}></TouchableOpacity>
          </View>
        </CameraView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 0,
  },
  button: {
    flex: 1,
    paddingTop: 10,
    paddingBottom: 100,
    paddingLeft: 100,
    paddingRight: 100,
    opacity: 100,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});