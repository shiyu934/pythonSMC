package com.csie.trading;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.stereotype.Service;
import java.io.*;
import java.net.*;
import java.util.concurrent.atomic.AtomicReference;

@Service
@Controller
public class TacticalCenter {

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    // Store the last INIT payload so new clients can receive it on reconnect
    private final AtomicReference<String> lastInitPayload = new AtomicReference<>(null);

    @EventListener(ApplicationReadyEvent.class)
    public void startBridge() {
        new Thread(() -> {
            try (ServerSocket server = new ServerSocket(9998)) {
                System.out.println("[OK] Java TacticalCenter ready on Port 9998, waiting for Python...");

                while (true) {
                    // 【關鍵修正】手動加大 Buffer 到 1024 * 1024 (1MB)
                    try (Socket socket = server.accept();
                        BufferedReader in = new BufferedReader(
                            new InputStreamReader(socket.getInputStream(), "UTF-8"), 
                            1024 * 1024)) {

                        System.out.println("[OK] Python bridge connected");
                        String jsonLine;

                        while ((jsonLine = in.readLine()) != null) {
                            // [DEBUG] 監控收到的封包長度
                            int len = jsonLine.length();
                            if (len > 0) {
                                // 如果收到很短的資料，可能是心跳包；如果很長，就是 K 線
                                System.out.println("[DEBUG] Data received, length: " + len);
                            }

                            // 核心判斷：快取歷史底圖
                            // 注意：如果 JSON 被截斷，這裡的 contains 就會失效
                            if (jsonLine.contains("\"type\":\"INIT\"")) {
                                lastInitPayload.set(jsonLine);
                                System.out.println("[OK] INIT payload cached! (Total size: " + len + " bytes)");
                            }

                            // 廣播給前端 (STOMP WebSocket)
                            messagingTemplate.convertAndSend("/topic/tactical", jsonLine);
                        }

                    } catch (Exception e) {
                        System.err.println("[WARN] Python connection error: " + e.getMessage());
                        // 這裡不需要 break，讓它回到 server.accept() 等待 Python 重連
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();
    }

    // Frontend calls /app/request-init after connecting to get the cached INIT payload
    // This ensures K-line history is available after page refresh
    @MessageMapping("/request-init")
    public void handleInitRequest() {
        String cached = lastInitPayload.get();
        if (cached != null) {
            System.out.println("[OK] Sending cached INIT to new client");
            messagingTemplate.convertAndSend("/topic/tactical", cached);
        } else {
            System.out.println("[WARN] No cached INIT yet, waiting for Python to send one");
        }
    }
}