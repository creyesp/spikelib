"""Implementing of classes and function to prepare raw data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuroshare as ns


class Sync:
    """Get the sinchronization times from a record.

    Class provide a complete enviroment to explore, cumpute and
    recovery sincronization signal in a record using a SamplingInteface
    script. SamplingInteface scripts send images to the projector and
    it use someone of RGB channel (in a VGA cable) to target the exact
    time when a frame was showed. That channel use different intensity
    levels to mark different frames.
    The following image show a simple example how to work it.
    https://tinyurl.com/ydxlvds9

    """

    def __init__(self, exp_name, real_fps=59.7596):
        """Initialization of Sync.

        Parameters
        ----------
        exp_name : str
            The name of the experiment
        real_fps : float
            The real refresh rate used by the projector. Please check
            log file to get this number.
        """
        self.exp_name = exp_name
        self.real_fps = real_fps

    def __repr__(self):
        return '{}({},{})'.format(self.__class__, self.exp_name, self.real_fps)

    def read_mcd(self, mcd_file):
        """Load mcd file to be analyze.

        Parameters
        ----------
        mcd_file : str
            Path to mcd file

        """
        self.data = ns.File(mcd_file)
        time_resolution = self.data.metadata_raw['TimeStampResolution']
        self.sample_rate = 1/time_resolution

    def show_entities(self):
        """Show all channel in mcd file."""
        print('Entities in MCD file.\nindex:  \t label:  \t entity_type: ')
        for kidx, entity in enumerate(self.data.entities):
            kvalues = [kidx, entity.label, entity.entity_type]
            print('{:04d}: {} type: {:d}'.format(*kvalues))

    def load_analyzed(self, src_folder):
        """Load output files from Analizer mathod.

        Load the result of analysed mcd file with analyzer method and
        saved previously. Those file have start_end points, repeated
        frame points and duration of analyzed experiment.

        Parameters
        ----------
        src_folder : str
            path of directory with result files.

        """
        file_name = src_folder + 'start_end_frames_' + self.exp_name + '.txt'
        self.start_end_frame = np.loadtxt(file_name, dtype='int32')
        file_name = src_folder + 'repeated_frames_' + self.exp_name + '.txt'
        self.repeted_start_frames = np.loadtxt(file_name, dtype='int32')
        file_name = src_folder + 'total_duration_' + self.exp_name + '.txt'
        general_information = np.loadtxt(file_name, dtype='int32')
        self.total_duration = general_information[0]
        self.sample_rate = general_information[1]

    def load_events(self, source_file):
        """Read csv file with event for a experiment.

        Update event_list atribute from csv file as dataframe.

        Parameters
        ----------
        source_file: str
            path csv file with list of event for a experiment.

        """
        self.event_list = pd.read_csv(source_file)

    def get_raw_data(self, channel, start=0, windows=-1):
        """Load all raw data in a specific channel.

        Parameters
        ----------
        channel : int
            number of channel to get raw data in mcd file.
        start : int
            Start point to get data.
        windows : int
            Number of point (window) after start point to get data.

        Return
        ----------
        entity_data : array
            Record values in a channel [samples].
        analog_time : array
            Time of Record values in a channel [s].
        dur : int
            Duration of record in a channel [samples]

        Example
        ----------
        data, time, dur = sync.get_raw_data(channel)

        """
        try:
            entity = self.data.entities[channel]
            entity_data, analog_time, dur = entity.get_data(start, windows)
            return entity_data, analog_time, dur
        except AttributeError as err:
            print(err, ', please load a mcd file.')

    def get_raw_samples(self, channel, start=0, windows=-1):
        """Load all raw data in a specific channel.

        Parameters
        ----------
        channel : int
            number of channel to get raw data in mcd file.
        start : int
            Start point to get data.
        windows : int
            Number of point (window) after start point to get data.

        Return
        ----------
        entity_data : array
            Record values in a channel [samples].
        dur : int
            Duration of record in a channel [samples]

        Example
        ----------
        sync = Sync('name')
        sync.read_mcd(mcd_path)
        data, dur = sync.get_raw_data(channel)

        """
        try:
            entity = self.data.entities[channel]
            entity_data, _, dur = entity.get_data(start, windows)
            return entity_data, dur
        except AttributeError as err:
            print(err, ', please load a mcd file.')

    def plot_window(self, channel, start_point, window):
        """Plot analog signal of syncronization in a specific time.

        Take a window time of the analog signal in MCD file and plot
        it to show the raw values.

        Parameters
        ----------
        channel : int
            number of channel to plot
        start_point : int
            start point to plot
        window : int
            number of points to plot

        """
        entity = self.data.entities[channel]
        analog_data, analog_time, dur = entity.get_data(start_point, window)
        plt.figure()
        plt.plot(analog_time-analog_time[0], analog_data)
        plt.show()

    def analyzer(self, channel):
        """Get the sinchronization times from MCD file.

        Recovery the exact time when a frame was showed  on the screen
        using one of VGA channels as trigger. This channel use
        different intensity levels to mark when a frame started. The
        following image show a simple example how to make it.
        https://tinyurl.com/y84n4xjh

        Parameters
        ----------
        mcd_channel : int
            channel number where is the analog signal in .mcd file. Use
            showEntities() to check number.

        Update
        ----------
        start_end_frames : attr <- int 2d array
            array with a start and end point for each frame in analog
            signal
        repeted_frames : attr <- int id array
            array points where the next frame it the repetition of this
        total_duration : attr <- int
            total number of points in record
        sample_rate : attr <- int
            sample rate of record

        ToDo
        ----------
        Detection method should be change for one more robust technique
        because now use search point to point where are the end of
        pulse. Try using a find_peak algorithm.

        """
        # Implementation
        # 1) each frame has a color bar for defaul in bottom of projector
        # 2) Analog signas has a pulse for each frame, the end of this pulse
        #    is when a frame was recieved by projector.
        # 3) this moment is the start of presentation frame
        # 4) the end of the last frame must be infer
        analog_data, analog_time, dur = self.get_raw_data(channel)
        # Threshold to delete the basal noise in volts.
        thresh_volts = 0.151111112

        # Number of points to consider a pulse as a valid frame
        wide_pulse = 50
        min_distanceFrames = np.floor(self.sample_rate/self.real_fps)
        max_distanceFrames = np.ceil(self.sample_rate/self.real_fps)

        # Get sync times
        filter_amp = analog_data > thresh_volts
        filter_amp[:wide_pulse] = False
        filter_amp[-2:] = False
        filter_amp_pos = np.where(filter_amp)[0]
        sync_point = []
        sync_amp = []
        for k in filter_amp_pos:
            before_point = analog_data[k-1] > thresh_volts
            after_point = analog_data[k+1] < thresh_volts
            before_pulse = analog_data[k-wide_pulse] < thresh_volts

            if before_point and before_pulse and after_point:
                sync_point.append(k)
                sync_amp.append(np.mean(analog_data[k-wide_pulse:k+1]))

        # Add start and end of data to find when occurred a event
        diff_frame = np.diff([0] + sync_point + [len(analog_data)])
        event_pos = np.where(diff_frame > max_distanceFrames)[0]
        end_event_pos = event_pos[1:] - 1

        # Start and end points
        start_frame = np.asarray(sync_point)
        end_frame = np.concatenate((sync_point[1:], sync_point[-1:]))

        # The finish time in a sequence of images is not register and
        # it's necesary to extrapolar this point
        end_frame[end_event_pos] = (start_frame[end_event_pos]
                                    + min_distanceFrames
                                    )
        end_frame[-1] = start_frame[-1] + min_distanceFrames

        sync_point = np.asarray(sync_point)
        self.start_end_frame = np.stack((start_frame, end_frame)).T

        # Detect when a frame was repeated
        thr_rep = 0.09
        amp_inter_frame = np.abs(np.diff(sync_amp))
        repeted_frame_point = np.where(amp_inter_frame < thr_rep)[0]
        self.repeted_start_frames = start_frame[repeted_frame_point]
        self.total_duration = len(analog_data)

    def create_events(self):
        """Create a list of events of sincronization.

        Use start time from each frame to detect a sequence of
        images (event) and create a dataframe with all event.
        This dataframe has start, end, duration, number of frame and
        time to next event.
        This image show a example of 2 events.
        https://tinyurl.com/ydxlvds9

        """
        start_frame = self.start_end_frame[:, 0]
        end_frame = self.start_end_frame[:, 1]
        diff_time = np.diff(np.concatenate(
            (np.array([0]), start_frame, np.array([self.total_duration]))))
        # Max dist between frames
        max_dist_frame = np.ceil(self.sample_rate/self.real_fps)
        filter_event = diff_time > max_dist_frame
        end_event_pos = np.where(filter_event)[0]

        # Select start and end time for each event
        start_event = start_frame[end_event_pos[:-1]]
        end_event = end_frame[end_event_pos[1:] - 1]
        n_frames = end_event_pos[1:] - end_event_pos[:-1]

        # Add bound condition
        start_event_full = np.concatenate(
            (np.array([0]), start_event, end_event[-1:]))
        end_event_full = np.concatenate(
            (start_event[:1], end_event, np.array([self.total_duration])))
        start_next_event = np.concatenate(
            (start_event_full[1:], end_event_full[-1:]))
        n_frames_full = np.concatenate(
            (np.array([0]), n_frames, np.array([0])))

        # Create a DataFrame to create a list of event
        event_list = pd.DataFrame({
            'start_event': start_event_full, 'end_event': end_event_full,
            'n_frames': n_frames_full, 'start_next_event': start_next_event})

        event_list = event_list[
            ['n_frames', 'start_event', 'end_event', 'start_next_event']
            ]
        event_list['event_duration'] = (
            event_list['end_event']
            - event_list['start_event']
            )
        event_list['event_duration_seg'] = (
            event_list['event_duration']
            / self.sample_rate
            )
        event_list['inter_event_duration'] = (
            event_list['start_next_event']
            - event_list['end_event']
            )
        event_list.loc[len(event_list)-1, 'inter_event_duration'] = 0
        event_list['inter_event_duration_seg'] = (
            event_list['inter_event_duration']
            / self.sample_rate
            )
        event_list['protocol_name'] = ''
        event_list['repetition_name'] = ''
        self.event_list = event_list

    def add_repeated(self):
        """Add repeated frames to event list.

        For each event in event_list attribute add the repeated frame
        points. This point is the right frame showed and next frame is
        the repeated frame.
        """
        self.event_list['repeated_frames'] = ''
        self.event_list['#repeated_frames'] = 0
        events = self.event_list[['start_event', 'end_event']].values
        for kidx, (kstart, kend) in enumerate(events):
            filter_rep = ((self.repeted_start_frames >= kstart)
                          * (self.repeted_start_frames <= kend)
                          )
            self.event_list.loc[kidx, 'repeated_frames'] = str(
                self.repeted_start_frames[filter_rep])
            self.event_list.loc[kidx, ['#repeated_frames']] = len(
                self.repeted_start_frames[filter_rep])

    def save_analyzed(self, output_folder, stype='txt'):
        """Save attributes generated by analyzer method.

        Save start_end frames, repeated frames, duration and sample
        rate of a record generated by analyzer method.

        Parameters
        ----------
        output_folder : str
            directory path to save files.
        stype : str
            type of output file. text file or hdf5 (Default txt)

        """
        if stype == 'txt':
            template_name = '{}_' + self.exp_name + '.txt'
            file_path = template_name.format('start_end_frames')
            np.savetxt(output_folder+file_path,
                       self.start_end_frame, fmt='%d',
                       header='start_frame [points], end_frame [points]')
            file_path = template_name.format('repeated_frames')
            np.savetxt(output_folder+file_path,
                       self.repeted_start_frames, fmt='%d',
                       header='repeated_frame [points]')
            file_path = template_name.format('total_duration')
            header_dur = 'total_duration [point], sample_rate [points]'
            array_dur = [[self.total_duration], [self.sample_rate]]
            np.savetxt(output_folder+file_path, np.array(array_dur).T,
                       fmt='%d', header=header_dur)
        elif stype == 'hdf5':
            pass

    def save_events(self, output_folder):
        """Save event list in a csv file.

        Take the event list dataframe and save it as csv.

        Parameters
        ----------
        output_folder : str
            directory path to save files.

        """
        template_name = output_folder + '{}_' + self.exp_name + '_.csv'
        self.event_list.to_csv(template_name.format('event_list'), index=False)

    def create_separated_sync(self, output_folder):
        """Split syncronization times for each event.

        Take start_end times of syncronization and split it in
        individual files.

        Parameters
        ----------
        output_folder : str
            directory path to save files.

        """
        event_bounds = self.event_list[
            ['start_event', 'start_next_event']].values
        pointer = self.start_end_frame

        for kidx, (kstart, kend) in enumerate(event_bounds):
            filter_frame = (pointer[:, 0] >= kstart)\
                           * (pointer[:, 0] < kend)
            start_end_event = pointer[filter_frame]
            np.savetxt(output_folder+'{:03d}.txt'.format(kidx),
                       start_end_event, fmt='%d')
            pointer = pointer[~filter_frame]

    def close_file(self):
        """Close mcd file."""
        if hasattr(self, 'data'):
            self.data.close()
        else:
            print('First need open a mcd file.')


class SyncDigital(Sync):
    """Get the sinchronization times from a digital record.

    Class provide a complete enviroment to explore, cumpute and
    recovery sincronization signal in a digital record.

    """
