FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV GZ_RENDERING_ENGINE=ogre
ENV DISPLAY=:1

RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release ca-certificates \
    tigervnc-standalone-server novnc websockify \
    libgl1-mesa-dri libgl1-mesa-glx libegl1-mesa mesa-utils \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
       -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
       http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

COPY start.sh /start.sh
COPY save_world.sh /usr/local/bin/save_world
RUN chmod +x /start.sh /usr/local/bin/save_world

EXPOSE 8080

CMD ["/start.sh"]
